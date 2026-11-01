/*
 * Review-response baseline: faithful TBE-like set-associative cache.
 *
 * This benchmark models the FBGEMM/TBE cache shape without depending on the
 * embedding-bag API:
 *   - 32-way set-associative cache by default.
 *   - 128B value rows by default (32 float values).
 *   - Staged lookup, miss compaction, victim selection/reservation,
 *     replacement/writeback, and value-returning gather.
 *   - LRU and LFU policies, with LRU as the primary reviewer-response policy.
 *
 * Output: CSV to stdout; progress/configuration to stderr.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "benchmark_util.cuh"

using K = uint64_t;
using Meta = unsigned long long;

enum class Policy { kLru, kLfu };

struct Config {
  std::string mode = "quick";
  std::string policy = "lru";
  std::vector<double> load_factors{0.25, 0.50, 0.75, 1.00};
  size_t capacity_slots = 1UL << 20;
  size_t batch_size = 1UL << 16;
  int associativity = 32;
  int value_bytes = 128;
  int warmup = 1;
  int runs = 2;
  double miss_rate = 0.25;
};

struct DeviceCache {
  K* keys = nullptr;
  int* valid = nullptr;
  float* values = nullptr;
  Meta* age = nullptr;
  Meta* freq = nullptr;
  int* locks = nullptr;

  void allocate(size_t capacity_slots, size_t value_dim, size_t num_sets) {
    CUDA_CHECK(cudaMalloc(&keys, capacity_slots * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&valid, capacity_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&values, capacity_slots * value_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&age, capacity_slots * sizeof(Meta)));
    CUDA_CHECK(cudaMalloc(&freq, capacity_slots * sizeof(Meta)));
    CUDA_CHECK(cudaMalloc(&locks, num_sets * sizeof(int)));
  }

  void free() {
    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(valid));
    CUDA_CHECK(cudaFree(values));
    CUDA_CHECK(cudaFree(age));
    CUDA_CHECK(cudaFree(freq));
    CUDA_CHECK(cudaFree(locks));
  }
};

struct RequestBuffers {
  K* keys = nullptr;
  float* incoming_values = nullptr;
  float* output_values = nullptr;
  float* writeback_values = nullptr;
  int* hit_flags = nullptr;
  int* slot_for_request = nullptr;
  int* miss_indices = nullptr;
  int* victim_slots = nullptr;
  int* evicted_valid = nullptr;
  K* evicted_keys = nullptr;
  unsigned long long* hit_count = nullptr;
  unsigned long long* miss_count = nullptr;
  unsigned long long* replacement_count = nullptr;
  unsigned long long* writeback_bytes = nullptr;
  unsigned long long* global_clock = nullptr;

  void allocate(size_t batch_size, size_t value_dim) {
    CUDA_CHECK(cudaMalloc(&keys, batch_size * sizeof(K)));
    CUDA_CHECK(
        cudaMalloc(&incoming_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&output_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&writeback_values, batch_size * value_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hit_flags, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&slot_for_request, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&miss_indices, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&victim_slots, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&evicted_valid, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&evicted_keys, batch_size * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&hit_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&miss_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&replacement_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&writeback_bytes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&global_clock, sizeof(unsigned long long)));
  }

  void free() {
    CUDA_CHECK(cudaFree(keys));
    CUDA_CHECK(cudaFree(incoming_values));
    CUDA_CHECK(cudaFree(output_values));
    CUDA_CHECK(cudaFree(writeback_values));
    CUDA_CHECK(cudaFree(hit_flags));
    CUDA_CHECK(cudaFree(slot_for_request));
    CUDA_CHECK(cudaFree(miss_indices));
    CUDA_CHECK(cudaFree(victim_slots));
    CUDA_CHECK(cudaFree(evicted_valid));
    CUDA_CHECK(cudaFree(evicted_keys));
    CUDA_CHECK(cudaFree(hit_count));
    CUDA_CHECK(cudaFree(miss_count));
    CUDA_CHECK(cudaFree(replacement_count));
    CUDA_CHECK(cudaFree(writeback_bytes));
    CUDA_CHECK(cudaFree(global_clock));
  }
};

struct Counters {
  unsigned long long hits = 0;
  unsigned long long misses = 0;
  unsigned long long replacements = 0;
  unsigned long long writeback_bytes = 0;
};

struct StageTimes {
  float lookup_ms = 0.0f;
  float miss_ms = 0.0f;
  float victim_ms = 0.0f;
  float replacement_ms = 0.0f;
  float return_ms = 0.0f;

  float total_ms() const {
    return lookup_ms + miss_ms + victim_ms + replacement_ms + return_ms;
  }
};

static size_t div_up(size_t n, size_t d) { return (n + d - 1) / d; }

static int grid_for(size_t n, int block) {
  size_t grid = div_up(n, static_cast<size_t>(block));
  return static_cast<int>(std::min<size_t>(grid, 65535));
}

static std::vector<double> parse_lf_list(const std::string& input) {
  std::vector<double> values;
  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::stod(item));
  }
  return values;
}

static std::string policy_name(Policy policy) {
  return policy == Policy::kLru ? "lru" : "lfu";
}

static void apply_mode_defaults(Config* cfg) {
  if (cfg->mode == "quick") {
    cfg->capacity_slots = 1UL << 20;
    cfg->batch_size = 1UL << 16;
    cfg->warmup = 1;
    cfg->runs = 2;
  } else if (cfg->mode == "full") {
    cfg->capacity_slots = 128UL * 1024 * 1024;
    cfg->batch_size = 1024UL * 1024;
    cfg->warmup = 3;
    cfg->runs = 5;
  } else {
    std::cerr << "Unknown mode: " << cfg->mode << std::endl;
    std::exit(1);
  }
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  bool capacity_overridden = false;
  bool batch_overridden = false;
  bool warmup_overridden = false;
  bool runs_overridden = false;

  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(1);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--mode") {
      cfg.mode = require_value("--mode");
    } else if (arg.rfind("--mode=", 0) == 0) {
      cfg.mode = arg.substr(7);
    } else if (arg == "--quick") {
      cfg.mode = "quick";
    } else if (arg == "--full") {
      cfg.mode = "full";
    } else if (arg == "--policy") {
      cfg.policy = require_value("--policy");
    } else if (arg.rfind("--policy=", 0) == 0) {
      cfg.policy = arg.substr(9);
    } else if (arg == "--lf") {
      cfg.load_factors = parse_lf_list(require_value("--lf"));
    } else if (arg.rfind("--lf=", 0) == 0) {
      cfg.load_factors = parse_lf_list(arg.substr(5));
    } else if (arg == "--capacity") {
      cfg.capacity_slots = std::stoull(require_value("--capacity"));
      capacity_overridden = true;
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity_slots = std::stoull(arg.substr(11));
      capacity_overridden = true;
    } else if (arg == "--batch") {
      cfg.batch_size = std::stoull(require_value("--batch"));
      batch_overridden = true;
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch_size = std::stoull(arg.substr(8));
      batch_overridden = true;
    } else if (arg == "--warmup") {
      cfg.warmup = std::stoi(require_value("--warmup"));
      warmup_overridden = true;
    } else if (arg.rfind("--warmup=", 0) == 0) {
      cfg.warmup = std::stoi(arg.substr(9));
      warmup_overridden = true;
    } else if (arg == "--runs") {
      cfg.runs = std::stoi(require_value("--runs"));
      runs_overridden = true;
    } else if (arg.rfind("--runs=", 0) == 0) {
      cfg.runs = std::stoi(arg.substr(7));
      runs_overridden = true;
    } else if (arg == "--assoc") {
      cfg.associativity = std::stoi(require_value("--assoc"));
    } else if (arg.rfind("--assoc=", 0) == 0) {
      cfg.associativity = std::stoi(arg.substr(8));
    } else if (arg == "--value-bytes") {
      cfg.value_bytes = std::stoi(require_value("--value-bytes"));
    } else if (arg.rfind("--value-bytes=", 0) == 0) {
      cfg.value_bytes = std::stoi(arg.substr(14));
    } else if (arg == "--miss-rate") {
      cfg.miss_rate = std::stod(require_value("--miss-rate"));
    } else if (arg.rfind("--miss-rate=", 0) == 0) {
      cfg.miss_rate = std::stod(arg.substr(12));
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: tbe_like_cache_baseline [options]\n"
                << "  --mode quick|full          Default: quick\n"
                << "  --policy lru|lfu|all       Default: lru\n"
                << "  --lf 0.25,0.50,0.75,1.00  Load factors to sweep\n"
                << "  --capacity N               Override cache slots\n"
                << "  --batch N                  Override request batch size\n"
                << "  --assoc N                  Default: 32\n"
                << "  --value-bytes N            Default: 128\n"
                << "  --miss-rate P              Default: 0.25\n"
                << "  --warmup N --runs N\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(1);
    }
  }

  Config mode_defaults;
  mode_defaults.mode = cfg.mode;
  apply_mode_defaults(&mode_defaults);
  if (!capacity_overridden) cfg.capacity_slots = mode_defaults.capacity_slots;
  if (!batch_overridden) cfg.batch_size = mode_defaults.batch_size;
  if (!warmup_overridden) cfg.warmup = mode_defaults.warmup;
  if (!runs_overridden) cfg.runs = mode_defaults.runs;

  if (cfg.associativity <= 0 || cfg.associativity > 128) {
    std::cerr << "Associativity must be in [1, 128]" << std::endl;
    std::exit(1);
  }
  if (cfg.value_bytes <= 0 ||
      cfg.value_bytes % static_cast<int>(sizeof(float))) {
    std::cerr << "value-bytes must be a positive multiple of sizeof(float)"
              << std::endl;
    std::exit(1);
  }
  if (cfg.capacity_slots < static_cast<size_t>(cfg.associativity) ||
      cfg.capacity_slots % static_cast<size_t>(cfg.associativity) != 0) {
    std::cerr << "capacity must be a multiple of associativity" << std::endl;
    std::exit(1);
  }
  if (cfg.miss_rate < 0.0 || cfg.miss_rate > 1.0) {
    std::cerr << "miss-rate must be in [0, 1]" << std::endl;
    std::exit(1);
  }
  if (cfg.load_factors.empty()) {
    std::cerr << "At least one load factor is required" << std::endl;
    std::exit(1);
  }
  for (double lf : cfg.load_factors) {
    if (lf <= 0.0 || lf > 1.0) {
      std::cerr << "Load factor must be in (0, 1]: " << lf << std::endl;
      std::exit(1);
    }
  }
  if (cfg.policy != "lru" && cfg.policy != "lfu" && cfg.policy != "all") {
    std::cerr << "policy must be lru, lfu, or all" << std::endl;
    std::exit(1);
  }
  return cfg;
}

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float deterministic_value(K key, int dim) {
  return static_cast<float>((key & 0xffffULL) * 0.001 + dim);
}

__global__ void init_cache_kernel(K* keys, int* valid, float* values, Meta* age,
                                  Meta* freq, int* locks, size_t capacity_slots,
                                  size_t num_sets, int associativity,
                                  int resident_ways, int value_dim) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t slot = tid; slot < capacity_slots; slot += stride) {
    size_t set = slot / associativity;
    int way = static_cast<int>(slot % associativity);
    int is_valid = way < resident_ways ? 1 : 0;
    K key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    keys[slot] = key;
    valid[slot] = is_valid;
    age[slot] = is_valid ? static_cast<Meta>(way + 1) : 0;
    freq[slot] = is_valid ? 1 : 0;
  }

  size_t total_values = capacity_slots * static_cast<size_t>(value_dim);
  for (size_t idx = tid; idx < total_values; idx += stride) {
    size_t slot = idx / value_dim;
    int dim = static_cast<int>(idx % value_dim);
    size_t set = slot / associativity;
    int way = static_cast<int>(slot % associativity);
    K key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    values[idx] = deterministic_value(key, dim);
  }

  for (size_t set = tid; set < num_sets; set += stride) {
    locks[set] = 0;
  }
}

__global__ void prepare_requests_kernel(K* request_keys, float* incoming_values,
                                        size_t batch_size, size_t num_sets,
                                        int resident_ways, int value_dim,
                                        double miss_rate, uint64_t seed,
                                        uint64_t miss_key_base) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    uint64_t r = mix64(seed ^ i);
    double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
    K key;
    if (u < miss_rate) {
      key = static_cast<K>(miss_key_base + i);
    } else {
      size_t set = mix64(seed + i * 17ULL) % num_sets;
      int way = static_cast<int>(mix64(seed + i * 31ULL) %
                                 static_cast<uint64_t>(resident_ways));
      key = static_cast<K>(set + static_cast<size_t>(way) * num_sets);
    }
    request_keys[i] = key;
    for (int dim = 0; dim < value_dim; dim++) {
      incoming_values[i * static_cast<size_t>(value_dim) + dim] =
          deterministic_value(key, dim);
    }
  }
}

__global__ void lookup_stage_kernel(const K* request_keys, K* cache_keys,
                                    const int* valid, Meta* age, Meta* freq,
                                    int* hit_flags, int* slot_for_request,
                                    unsigned long long* hit_count,
                                    unsigned long long* global_clock,
                                    size_t batch_size, size_t num_sets,
                                    int associativity, int policy) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    K key = request_keys[i];
    size_t set = key % num_sets;
    size_t base = set * static_cast<size_t>(associativity);
    int found_slot = -1;

    for (int way = 0; way < associativity; way++) {
      size_t slot = base + static_cast<size_t>(way);
      if (valid[slot] && cache_keys[slot] == key) {
        found_slot = static_cast<int>(slot);
        break;
      }
    }

    hit_flags[i] = found_slot >= 0 ? 1 : 0;
    slot_for_request[i] = found_slot;
    if (found_slot >= 0) {
      atomicAdd(hit_count, 1ULL);
      if (policy == 0) {
        age[found_slot] = atomicAdd(global_clock, 1ULL) + 1ULL;
      } else {
        atomicAdd(&freq[found_slot], 1ULL);
        age[found_slot] = atomicAdd(global_clock, 1ULL) + 1ULL;
      }
    }
  }
}

__global__ void miss_compact_stage_kernel(const int* hit_flags,
                                          int* miss_indices,
                                          unsigned long long* miss_count,
                                          size_t batch_size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < batch_size; i += stride) {
    if (!hit_flags[i]) {
      unsigned long long pos = atomicAdd(miss_count, 1ULL);
      miss_indices[pos] = static_cast<int>(i);
    }
  }
}

__device__ __forceinline__ void lock_set(int* locks, size_t set) {
  while (atomicCAS(&locks[set], 0, 1) != 0) {
  }
}

__device__ __forceinline__ void unlock_set(int* locks, size_t set) {
  __threadfence();
  atomicExch(&locks[set], 0);
}

__global__ void victim_stage_kernel(
    const K* request_keys, K* cache_keys, int* valid, Meta* age, Meta* freq,
    int* locks, const int* miss_indices, int* victim_slots, int* evicted_valid,
    K* evicted_keys, int* hit_flags, int* slot_for_request,
    const unsigned long long* miss_count, unsigned long long* replacement_count,
    unsigned long long* writeback_bytes, unsigned long long* global_clock,
    size_t num_sets, int associativity, int policy, int value_bytes) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  unsigned long long total_misses = *miss_count;

  for (unsigned long long pos = tid; pos < total_misses; pos += stride) {
    int request_idx = miss_indices[pos];
    K key = request_keys[request_idx];
    size_t set = key % num_sets;
    size_t base = set * static_cast<size_t>(associativity);

    lock_set(locks, set);

    int existing_slot = -1;
    for (int way = 0; way < associativity; way++) {
      size_t slot = base + static_cast<size_t>(way);
      if (valid[slot] && cache_keys[slot] == key) {
        existing_slot = static_cast<int>(slot);
        break;
      }
    }

    if (existing_slot >= 0) {
      victim_slots[pos] = -1;
      evicted_valid[pos] = 0;
      hit_flags[request_idx] = 1;
      slot_for_request[request_idx] = existing_slot;
      if (policy == 0) {
        age[existing_slot] = atomicAdd(global_clock, 1ULL) + 1ULL;
      } else {
        atomicAdd(&freq[existing_slot], 1ULL);
        age[existing_slot] = atomicAdd(global_clock, 1ULL) + 1ULL;
      }
      unlock_set(locks, set);
      continue;
    }

    int victim = -1;
    for (int way = 0; way < associativity; way++) {
      size_t slot = base + static_cast<size_t>(way);
      if (!valid[slot]) {
        victim = static_cast<int>(slot);
        break;
      }
    }

    if (victim < 0) {
      victim = static_cast<int>(base);
      for (int way = 1; way < associativity; way++) {
        size_t slot = base + static_cast<size_t>(way);
        if (policy == 0) {
          if (age[slot] < age[victim]) victim = static_cast<int>(slot);
        } else {
          if (freq[slot] < freq[victim] ||
              (freq[slot] == freq[victim] && age[slot] < age[victim])) {
            victim = static_cast<int>(slot);
          }
        }
      }
    }

    victim_slots[pos] = victim;
    evicted_valid[pos] = valid[victim];
    evicted_keys[pos] = cache_keys[victim];
    if (valid[victim]) {
      atomicAdd(writeback_bytes, static_cast<unsigned long long>(value_bytes));
    }

    cache_keys[victim] = key;
    valid[victim] = 1;
    age[victim] = atomicAdd(global_clock, 1ULL) + 1ULL;
    freq[victim] = 1;
    slot_for_request[request_idx] = victim;
    atomicAdd(replacement_count, 1ULL);

    unlock_set(locks, set);
  }
}

__global__ void replacement_stage_kernel(
    const float* incoming_values, float* cache_values, float* writeback_values,
    const int* miss_indices, const int* victim_slots, const int* evicted_valid,
    const unsigned long long* miss_count, int value_dim) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  unsigned long long total_values =
      (*miss_count) * static_cast<unsigned long long>(value_dim);

  for (unsigned long long idx = tid; idx < total_values; idx += stride) {
    unsigned long long pos = idx / static_cast<unsigned long long>(value_dim);
    int dim =
        static_cast<int>(idx % static_cast<unsigned long long>(value_dim));
    int victim = victim_slots[pos];
    if (victim < 0) continue;

    int request_idx = miss_indices[pos];
    size_t cache_idx = static_cast<size_t>(victim) * value_dim + dim;
    size_t request_value_idx =
        static_cast<size_t>(request_idx) * value_dim + dim;
    size_t writeback_idx = static_cast<size_t>(pos) * value_dim + dim;
    if (evicted_valid[pos]) {
      writeback_values[writeback_idx] = cache_values[cache_idx];
    }
    cache_values[cache_idx] = incoming_values[request_value_idx];
  }
}

__global__ void value_return_stage_kernel(const int* slot_for_request,
                                          const float* cache_values,
                                          float* output_values,
                                          size_t batch_size, int value_dim) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total_values = batch_size * static_cast<size_t>(value_dim);

  for (size_t idx = tid; idx < total_values; idx += stride) {
    size_t row = idx / value_dim;
    int dim = static_cast<int>(idx % value_dim);
    int slot = slot_for_request[row];
    output_values[idx] =
        slot >= 0 ? cache_values[static_cast<size_t>(slot) * value_dim + dim]
                  : 0.0f;
  }
}

static float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

static StageTimes run_once(const Config& cfg, Policy policy, DeviceCache* cache,
                           RequestBuffers* req, size_t num_sets,
                           int resident_ways, int value_dim, int run_index) {
  const int block = 256;
  const int request_grid = grid_for(cfg.batch_size, block);
  const int value_grid =
      grid_for(cfg.batch_size * static_cast<size_t>(value_dim), block);
  uint64_t seed = 0x5eed1234ULL + static_cast<uint64_t>(run_index) * 101ULL;
  uint64_t miss_key_base = static_cast<uint64_t>(cfg.capacity_slots) *
                               static_cast<uint64_t>(run_index + 2) +
                           1ULL;

  CUDA_CHECK(cudaMemset(req->hit_count, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(req->miss_count, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(req->replacement_count, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(req->writeback_bytes, 0, sizeof(unsigned long long)));

  prepare_requests_kernel<<<request_grid, block>>>(
      req->keys, req->incoming_values, cfg.batch_size, num_sets, resident_ways,
      value_dim, cfg.miss_rate, seed, miss_key_base);
  CUDA_CHECK(cudaGetLastError());

  StageTimes times;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  lookup_stage_kernel<<<request_grid, block>>>(
      req->keys, cache->keys, cache->valid, cache->age, cache->freq,
      req->hit_flags, req->slot_for_request, req->hit_count, req->global_clock,
      cfg.batch_size, num_sets, cfg.associativity,
      policy == Policy::kLru ? 0 : 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  times.lookup_ms = elapsed(start, stop);

  CUDA_CHECK(cudaEventRecord(start));
  miss_compact_stage_kernel<<<request_grid, block>>>(
      req->hit_flags, req->miss_indices, req->miss_count, cfg.batch_size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  times.miss_ms = elapsed(start, stop);

  CUDA_CHECK(cudaEventRecord(start));
  victim_stage_kernel<<<request_grid, block>>>(
      req->keys, cache->keys, cache->valid, cache->age, cache->freq,
      cache->locks, req->miss_indices, req->victim_slots, req->evicted_valid,
      req->evicted_keys, req->hit_flags, req->slot_for_request, req->miss_count,
      req->replacement_count, req->writeback_bytes, req->global_clock, num_sets,
      cfg.associativity, policy == Policy::kLru ? 0 : 1, cfg.value_bytes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  times.victim_ms = elapsed(start, stop);

  CUDA_CHECK(cudaEventRecord(start));
  replacement_stage_kernel<<<value_grid, block>>>(
      req->incoming_values, cache->values, req->writeback_values,
      req->miss_indices, req->victim_slots, req->evicted_valid, req->miss_count,
      value_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  times.replacement_ms = elapsed(start, stop);

  CUDA_CHECK(cudaEventRecord(start));
  value_return_stage_kernel<<<value_grid, block>>>(
      req->slot_for_request, cache->values, req->output_values, cfg.batch_size,
      value_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  times.return_ms = elapsed(start, stop);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return times;
}

static Counters read_counters(const RequestBuffers& req) {
  Counters c;
  CUDA_CHECK(cudaMemcpy(&c.hits, req.hit_count, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.misses, req.miss_count, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.replacements, req.replacement_count,
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&c.writeback_bytes, req.writeback_bytes,
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  return c;
}

static void emit_row(const std::string& mode, Policy policy, double lf,
                     const Config& cfg, size_t resident_slots, int run,
                     const std::string& stage, float elapsed_ms,
                     const Counters& counters) {
  double seconds = static_cast<double>(elapsed_ms) / 1000.0;
  double throughput = seconds > 0.0 ? static_cast<double>(cfg.batch_size) /
                                          seconds / (1024.0 * 1024.0 * 1024.0)
                                    : 0.0;
  std::cout << "TBE_like," << policy_name(policy) << "," << mode << ","
            << std::fixed << std::setprecision(2) << lf << ","
            << cfg.associativity << "," << cfg.value_bytes << ","
            << cfg.capacity_slots << "," << resident_slots << ","
            << cfg.batch_size << "," << run << "," << stage << ","
            << std::setprecision(6) << elapsed_ms << "," << throughput << ","
            << counters.hits << "," << counters.misses << ","
            << counters.replacements << "," << counters.writeback_bytes
            << std::endl;
}

static void run_policy_lf(const Config& cfg, Policy policy, double lf) {
  int value_dim = cfg.value_bytes / static_cast<int>(sizeof(float));
  size_t num_sets = cfg.capacity_slots / cfg.associativity;
  int resident_ways =
      std::max(1, static_cast<int>(std::llround(cfg.associativity * lf)));
  resident_ways = std::min(resident_ways, cfg.associativity);
  size_t resident_slots = num_sets * static_cast<size_t>(resident_ways);
  double actual_lf = static_cast<double>(resident_slots) /
                     static_cast<double>(cfg.capacity_slots);

  std::cerr << "[TBE-like] policy=" << policy_name(policy)
            << " mode=" << cfg.mode << " LF=" << std::fixed
            << std::setprecision(2) << actual_lf
            << " assoc=" << cfg.associativity
            << " value_bytes=" << cfg.value_bytes
            << " capacity=" << cfg.capacity_slots << " batch=" << cfg.batch_size
            << std::endl;

  DeviceCache cache;
  RequestBuffers req;
  cache.allocate(cfg.capacity_slots, value_dim, num_sets);
  req.allocate(cfg.batch_size, value_dim);

  const int block = 256;
  int init_grid =
      grid_for(std::max(cfg.capacity_slots,
                        cfg.capacity_slots * static_cast<size_t>(value_dim)),
               block);
  init_cache_kernel<<<init_grid, block>>>(
      cache.keys, cache.valid, cache.values, cache.age, cache.freq, cache.locks,
      cfg.capacity_slots, num_sets, cfg.associativity, resident_ways,
      value_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(req.global_clock, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int run = 0; run < cfg.warmup + cfg.runs; run++) {
    StageTimes times = run_once(cfg, policy, &cache, &req, num_sets,
                                resident_ways, value_dim, run);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (run >= cfg.warmup) {
      Counters counters = read_counters(req);
      int visible_run = run - cfg.warmup + 1;
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "lookup", times.lookup_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "miss_compact", times.miss_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "victim", times.victim_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "replacement", times.replacement_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "value_return", times.return_ms, counters);
      emit_row(cfg.mode, policy, actual_lf, cfg, resident_slots, visible_run,
               "total", times.total_ms(), counters);
    }
  }

  req.free();
  cache.free();
}

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review-response TBE-like baseline" << std::endl;
  std::cerr << "Default anchor: 32-way set-associative, 128B rows, staged "
               "lookup/miss/victim/replacement/value-returning"
            << std::endl;

  std::cout << "baseline,policy,mode,load_factor,associativity,value_bytes,"
               "capacity_slots,resident_slots,batch_size,run,stage,elapsed_ms,"
               "throughput_bkvs,hits,misses,replacements,writeback_bytes"
            << std::endl;

  std::vector<Policy> policies;
  if (cfg.policy == "all") {
    policies = {Policy::kLru, Policy::kLfu};
  } else if (cfg.policy == "lfu") {
    policies = {Policy::kLfu};
  } else {
    policies = {Policy::kLru};
  }

  for (Policy policy : policies) {
    for (double lf : cfg.load_factors) {
      run_policy_lf(cfg, policy, lf);
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
