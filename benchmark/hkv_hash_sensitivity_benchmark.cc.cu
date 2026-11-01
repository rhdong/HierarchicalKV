/*
 * HKV-level hash sensitivity benchmark for review-response experiments.
 *
 * Unlike hash_sensitivity_benchmark.cc.cu, this target exercises real HKV
 * kernels.  The runner rebuilds this binary with HKV_HASH_VARIANT set to
 * Murmur3, SplitMix64-style, xxHash-style avalanche, or wyhash-style finalizer.
 *
 * Output CSV:
 *   hash_variant,distribution,table_mode,capacity,batch,attempted,final_size,
 *   achieved_lf,first_eviction_lf,old_found_rate,recent_found_rate,
 *   insert_throughput_bkvs,find_throughput_bkvs,primary_bucket_cv,
 *   primary_bucket_p99,primary_bucket_max,notes
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"

namespace {

using K = uint64_t;
using V = float;
using S = uint64_t;
using Table =
    nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kCustomized>;

constexpr size_t kDim = 32;
constexpr size_t kBucketSize = 128;

enum class Distribution {
  kSequential,
  kUniform64,
  kStridedLowbits,
  kZipfUnique,
};

struct Config {
  std::string mode = "quick";
  size_t capacity = 128UL * 1024;
  size_t batch = 32UL * 1024;
  size_t query = 32UL * 1024;
  double overfill = 1.125;
  std::vector<Distribution> distributions;
  std::vector<nv::merlin::TableMode> table_modes;
};

struct BucketStats {
  double cv = 0.0;
  double p99 = 0.0;
  uint32_t max = 0;
};

struct Result {
  std::string distribution;
  std::string table_mode;
  size_t capacity = 0;
  size_t batch = 0;
  size_t attempted = 0;
  size_t final_size = 0;
  double achieved_lf = 0.0;
  double first_eviction_lf = 1.0;
  double old_found_rate = 0.0;
  double recent_found_rate = 0.0;
  double insert_throughput_bkvs = 0.0;
  double find_throughput_bkvs = 0.0;
  BucketStats primary_bucket_stats;
  std::string notes;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  void copy_from_host(const std::vector<T>& host, size_t count,
                      cudaStream_t stream = 0) const {
    if (count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(ptr_, host.data(), count * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

uint64_t splitmix64_host(uint64_t key) {
  uint64_t z = key + UINT64_C(0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

const char* distribution_name(Distribution distribution) {
  switch (distribution) {
    case Distribution::kSequential:
      return "sequential";
    case Distribution::kUniform64:
      return "uniform64";
    case Distribution::kStridedLowbits:
      return "strided_lowbits";
    case Distribution::kZipfUnique:
      return "zipf_unique";
  }
  return "unknown";
}

const char* table_mode_name(nv::merlin::TableMode mode) {
  return mode == nv::merlin::TableMode::kMemory ? "memory" : "throughput";
}

K sanitize_key(uint64_t key, uint64_t salt) {
  constexpr uint64_t kEmpty = nv::merlin::DEFAULT_EMPTY_KEY;
  constexpr uint64_t kReclaim = nv::merlin::DEFAULT_RECLAIM_KEY;
  if (key == kEmpty || key == kReclaim || key == 0) {
    key ^= UINT64_C(0x9e3779b97f4a7c15) + salt;
  }
  if (key == kEmpty || key == kReclaim || key == 0) {
    key = salt + 1;
  }
  return key;
}

class ZipfGenerator {
 public:
  ZipfGenerator(uint64_t n, double theta, uint64_t seed)
      : n_(n), theta_(theta), rng_(seed) {
    zeta_n_ = zeta(n_, theta_);
    zeta_2_ = zeta(2, theta_);
    alpha_ = 1.0 / (1.0 - theta_);
    eta_ = (1.0 - std::pow(2.0 / n_, 1.0 - theta_)) / (1.0 - zeta_2_ / zeta_n_);
  }

  uint64_t next() {
    double u = dist_(rng_);
    double uz = u * zeta_n_;
    if (uz < 1.0) return 0;
    if (uz < 1.0 + std::pow(0.5, theta_)) return 1;
    uint64_t val =
        static_cast<uint64_t>(n_ * std::pow(eta_ * u - eta_ + 1.0, alpha_));
    return std::min(val, n_ - 1);
  }

 private:
  static double zeta(uint64_t n, double theta) {
    constexpr uint64_t kExact = 10000;
    double sum = 0.0;
    uint64_t exact = std::min(n, kExact);
    for (uint64_t i = 1; i <= exact; i++) {
      sum += 1.0 / std::pow(static_cast<double>(i), theta);
    }
    if (n > exact && theta != 1.0) {
      sum += (std::pow(static_cast<double>(n), 1.0 - theta) -
              std::pow(static_cast<double>(exact), 1.0 - theta)) /
             (1.0 - theta);
    }
    return sum;
  }

  uint64_t n_;
  double theta_;
  double zeta_n_ = 0.0;
  double zeta_2_ = 0.0;
  double alpha_ = 0.0;
  double eta_ = 0.0;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

std::vector<K> make_keys(Distribution distribution, size_t offset, size_t count,
                         size_t key_range) {
  std::vector<K> keys;
  keys.reserve(count);

  if (distribution == Distribution::kZipfUnique) {
    ZipfGenerator zipf(std::max<size_t>(key_range, count * 8), 1.1,
                       0xBAD5EEDULL + offset);
    std::unordered_set<K> seen;
    seen.reserve(count * 2 + 1);
    while (keys.size() < count) {
      uint64_t rank = zipf.next();
      K key = sanitize_key(rank + 1, offset + keys.size());
      if (seen.insert(key).second) keys.push_back(key);
    }
    return keys;
  }

  for (size_t i = 0; i < count; i++) {
    uint64_t index = offset + i;
    uint64_t key = 0;
    switch (distribution) {
      case Distribution::kSequential:
        key = index + 1;
        break;
      case Distribution::kUniform64:
        key = splitmix64_host(index + UINT64_C(0xD1B54A32D192ED03));
        break;
      case Distribution::kStridedLowbits:
        key = (index + 1) << 16;
        break;
      case Distribution::kZipfUnique:
        break;
    }
    keys.push_back(sanitize_key(key, index + 1));
  }
  return keys;
}

std::vector<S> make_scores(size_t offset, size_t count) {
  std::vector<S> scores(count);
  for (size_t i = 0; i < count; i++) {
    scores[i] = static_cast<S>(offset + i + 1);
  }
  return scores;
}

BucketStats compute_primary_bucket_stats(const std::vector<K>& keys,
                                         size_t capacity,
                                         nv::merlin::TableMode mode) {
  size_t bucket_count = std::max<size_t>(1, capacity / kBucketSize);
  std::vector<uint32_t> occupancy(bucket_count, 0);

  for (K key : keys) {
    uint64_t hash = nv::merlin::Murmur3HashHost(key);
    size_t bucket = 0;
    if (mode == nv::merlin::TableMode::kMemory) {
      bucket = static_cast<uint32_t>(hash) % bucket_count;
    } else {
      size_t global_idx = static_cast<size_t>(hash % capacity);
      bucket = global_idx / kBucketSize;
    }
    occupancy[bucket]++;
  }

  std::vector<uint32_t> sorted = occupancy;
  std::sort(sorted.begin(), sorted.end());

  double mean = 0.0;
  for (uint32_t value : occupancy) mean += value;
  mean /= static_cast<double>(occupancy.size());

  double variance = 0.0;
  for (uint32_t value : occupancy) {
    double delta = static_cast<double>(value) - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(occupancy.size());

  BucketStats stats;
  stats.cv = mean == 0.0 ? 0.0 : std::sqrt(variance) / mean;
  size_t p99_index =
      std::min(sorted.size() - 1,
               static_cast<size_t>(std::ceil(sorted.size() * 0.99)) - 1);
  stats.p99 = static_cast<double>(sorted[p99_index]);
  stats.max = sorted.back();
  return stats;
}

double found_rate(Table& table, const std::vector<K>& keys,
                  DeviceBuffer<K>& d_keys, DeviceBuffer<V>& d_values,
                  DeviceBuffer<bool>& d_found, cudaStream_t stream,
                  double* throughput_bkvs = nullptr) {
  if (keys.empty()) return 0.0;

  d_keys.copy_from_host(keys, keys.size(), stream);
  CUDA_CHECK(
      cudaMemsetAsync(d_found.get(), 0, keys.size() * sizeof(bool), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto timer = benchmark::KernelTimer<double>();
  timer.start();
  table.find(keys.size(), d_keys.get(), d_values.get(), d_found.get(), nullptr,
             stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.end();

  std::vector<uint8_t> found(keys.size());
  CUDA_CHECK(cudaMemcpyAsync(found.data(), d_found.get(),
                             keys.size() * sizeof(bool), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  size_t count = 0;
  for (uint8_t flag : found) count += flag ? 1 : 0;
  if (throughput_bkvs != nullptr) {
    *throughput_bkvs =
        keys.size() / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
  }
  return static_cast<double>(count) / static_cast<double>(keys.size());
}

void init_table(Table& table, const Config& cfg, nv::merlin::TableMode mode) {
  nv::merlin::HashTableOptions options;
  options.init_capacity = cfg.capacity;
  options.max_capacity = cfg.capacity;
  options.dim = kDim;
  options.max_bucket_size = kBucketSize;
  options.max_load_factor = 1.0f;
  options.max_hbm_for_vectors = cfg.capacity * kDim * sizeof(V);
  options.table_mode = mode;
  options.api_lock = true;

  table.init(options);
}

Result run_case(const Config& cfg, Distribution distribution,
                nv::merlin::TableMode mode, cudaStream_t stream) {
  Table table;
  init_table(table, cfg, mode);
  const size_t max_attempted =
      static_cast<size_t>(std::ceil(cfg.capacity * cfg.overfill));
  const size_t key_range =
      std::max<size_t>(cfg.capacity * 64, max_attempted * 4);

  DeviceBuffer<K> d_keys(cfg.batch);
  DeviceBuffer<V> d_values(cfg.batch * kDim);
  DeviceBuffer<S> d_scores(cfg.batch);
  std::vector<V> values(cfg.batch * kDim, 1.0f);

  std::vector<K> first_keys;
  std::vector<K> recent_keys;
  first_keys.reserve(std::min(cfg.query, cfg.batch));
  recent_keys.reserve(cfg.query);

  size_t attempted = 0;
  size_t prev_size = 0;
  double first_eviction_lf = 1.0;
  bool saw_eviction = false;
  double insert_seconds = 0.0;

  while (attempted < max_attempted) {
    size_t cur = std::min(cfg.batch, max_attempted - attempted);
    std::vector<K> keys = make_keys(distribution, attempted, cur, key_range);
    std::vector<S> scores = make_scores(attempted, cur);

    if (first_keys.size() < cfg.query) {
      size_t take = std::min(cfg.query - first_keys.size(), keys.size());
      first_keys.insert(first_keys.end(), keys.begin(), keys.begin() + take);
    }
    recent_keys.insert(recent_keys.end(), keys.begin(), keys.end());
    if (recent_keys.size() > cfg.query) {
      recent_keys.erase(recent_keys.begin(),
                        recent_keys.begin() + (recent_keys.size() - cfg.query));
    }

    d_keys.copy_from_host(keys, cur, stream);
    d_values.copy_from_host(values, cur * kDim, stream);
    d_scores.copy_from_host(scores, cur, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto timer = benchmark::KernelTimer<double>();
    timer.start();
    table.insert_or_assign(cur, d_keys.get(), d_values.get(), d_scores.get(),
                           stream, true, false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    timer.end();
    insert_seconds += timer.getResult();

    attempted += cur;
    size_t current_size = table.size(stream);
    if (!saw_eviction && current_size < attempted) {
      saw_eviction = true;
      first_eviction_lf =
          static_cast<double>(prev_size) / static_cast<double>(cfg.capacity);
    }
    prev_size = current_size;
  }

  DeviceBuffer<K> d_find_keys(cfg.query);
  DeviceBuffer<V> d_find_values(cfg.query * kDim);
  DeviceBuffer<bool> d_found(cfg.query);
  double find_tp = 0.0;
  double old_found = found_rate(table, first_keys, d_find_keys, d_find_values,
                                d_found, stream);
  double recent_found = found_rate(table, recent_keys, d_find_keys,
                                   d_find_values, d_found, stream, &find_tp);

  std::vector<K> sample_for_buckets = make_keys(
      distribution, 0, std::min(max_attempted, cfg.capacity), key_range);

  Result result;
  result.distribution = distribution_name(distribution);
  result.table_mode = table_mode_name(mode);
  result.capacity = cfg.capacity;
  result.batch = cfg.batch;
  result.attempted = attempted;
  result.final_size = table.size(stream);
  result.achieved_lf = static_cast<double>(result.final_size) /
                       static_cast<double>(cfg.capacity);
  result.first_eviction_lf = first_eviction_lf;
  result.old_found_rate = old_found;
  result.recent_found_rate = recent_found;
  result.insert_throughput_bkvs =
      attempted / insert_seconds / (1024.0 * 1024.0 * 1024.0);
  result.find_throughput_bkvs = find_tp;
  result.primary_bucket_stats =
      compute_primary_bucket_stats(sample_for_buckets, cfg.capacity, mode);
  result.notes = saw_eviction ? "eviction_detected" : "no_eviction_detected";
  return result;
}

void print_header() {
  std::cout
      << "hash_variant,distribution,table_mode,capacity,batch,attempted,"
         "final_size,achieved_lf,first_eviction_lf,old_found_rate,"
         "recent_found_rate,insert_throughput_bkvs,find_throughput_bkvs,"
         "primary_bucket_cv,primary_bucket_p99,primary_bucket_max,notes\n";
}

void print_result(const Result& result) {
  std::cout << nv::merlin::HkvHashVariantNameHost() << ','
            << result.distribution << ',' << result.table_mode << ','
            << result.capacity << ',' << result.batch << ',' << result.attempted
            << ',' << result.final_size << ',' << std::fixed
            << std::setprecision(6) << result.achieved_lf << ','
            << result.first_eviction_lf << ',' << result.old_found_rate << ','
            << result.recent_found_rate << ',' << result.insert_throughput_bkvs
            << ',' << result.find_throughput_bkvs << ','
            << result.primary_bucket_stats.cv << ','
            << result.primary_bucket_stats.p99 << ','
            << result.primary_bucket_stats.max << ',' << result.notes << '\n';
}

Distribution parse_distribution(const std::string& value) {
  if (value == "sequential") return Distribution::kSequential;
  if (value == "uniform64") return Distribution::kUniform64;
  if (value == "strided_lowbits") return Distribution::kStridedLowbits;
  if (value == "zipf_unique") return Distribution::kZipfUnique;
  throw std::invalid_argument("unknown distribution: " + value);
}

nv::merlin::TableMode parse_table_mode(const std::string& value) {
  if (value == "throughput") return nv::merlin::TableMode::kThroughput;
  if (value == "memory") return nv::merlin::TableMode::kMemory;
  throw std::invalid_argument("unknown table mode: " + value);
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) tokens.push_back(token);
  }
  return tokens;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto next_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + name);
      return argv[++i];
    };

    if (arg == "--mode") {
      cfg.mode = next_value(arg);
    } else if (arg.rfind("--mode=", 0) == 0) {
      cfg.mode = arg.substr(7);
    } else if (arg == "--capacity") {
      cfg.capacity = std::stoull(next_value(arg));
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity = std::stoull(arg.substr(11));
    } else if (arg == "--batch") {
      cfg.batch = std::stoull(next_value(arg));
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch = std::stoull(arg.substr(8));
    } else if (arg == "--query") {
      cfg.query = std::stoull(next_value(arg));
    } else if (arg.rfind("--query=", 0) == 0) {
      cfg.query = std::stoull(arg.substr(8));
    } else if (arg == "--overfill") {
      cfg.overfill = std::stod(next_value(arg));
    } else if (arg.rfind("--overfill=", 0) == 0) {
      cfg.overfill = std::stod(arg.substr(11));
    } else if (arg == "--distributions") {
      cfg.distributions.clear();
      for (const auto& value : split_csv(next_value(arg))) {
        cfg.distributions.push_back(parse_distribution(value));
      }
    } else if (arg.rfind("--distributions=", 0) == 0) {
      cfg.distributions.clear();
      for (const auto& value : split_csv(arg.substr(16))) {
        cfg.distributions.push_back(parse_distribution(value));
      }
    } else if (arg == "--table_modes") {
      cfg.table_modes.clear();
      for (const auto& value : split_csv(next_value(arg))) {
        cfg.table_modes.push_back(parse_table_mode(value));
      }
    } else if (arg.rfind("--table_modes=", 0) == 0) {
      cfg.table_modes.clear();
      for (const auto& value : split_csv(arg.substr(14))) {
        cfg.table_modes.push_back(parse_table_mode(value));
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: hkv_hash_sensitivity_benchmark [--mode quick|full]\n"
                << "       [--capacity=N] [--batch=N] [--query=N]\n"
                << "       [--overfill=RATIO]\n"
                << "       "
                   "[--distributions=sequential,uniform64,strided_lowbits,zipf_"
                   "unique]\n"
                << "       [--table_modes=throughput,memory]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (cfg.mode == "full") {
    cfg.capacity = 1024UL * 1024;
    cfg.batch = 128UL * 1024;
    cfg.query = 64UL * 1024;
    cfg.overfill = 1.125;
    cfg.distributions = {Distribution::kSequential, Distribution::kUniform64,
                         Distribution::kStridedLowbits,
                         Distribution::kZipfUnique};
    cfg.table_modes = {nv::merlin::TableMode::kThroughput,
                       nv::merlin::TableMode::kMemory};
  } else if (cfg.distributions.empty()) {
    cfg.distributions = {Distribution::kSequential,
                         Distribution::kStridedLowbits};
    cfg.table_modes = {nv::merlin::TableMode::kThroughput,
                       nv::merlin::TableMode::kMemory};
  }

  if (cfg.capacity % kBucketSize != 0) {
    throw std::invalid_argument("capacity must be divisible by bucket size");
  }
  cfg.batch = std::min(cfg.batch, cfg.capacity);
  cfg.query = std::min(cfg.query, cfg.batch);
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cerr << "GPU: " << props.name << "\n";
    std::cerr << "HKV hash variant: " << nv::merlin::HkvHashVariantNameHost()
              << " (" << HKV_HASH_VARIANT << ")\n";
    std::cerr << "mode=" << cfg.mode << " capacity=" << cfg.capacity
              << " batch=" << cfg.batch << " query=" << cfg.query
              << " overfill=" << cfg.overfill << "\n";

    CUDA_CHECK(cudaFree(0));
    print_header();
    for (Distribution distribution : cfg.distributions) {
      for (auto table_mode : cfg.table_modes) {
        std::cerr << "running distribution=" << distribution_name(distribution)
                  << " table_mode=" << table_mode_name(table_mode) << std::endl;
        print_result(run_case(cfg, distribution, table_mode, 0));
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    std::cerr << "hkv_hash_sensitivity_benchmark: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
