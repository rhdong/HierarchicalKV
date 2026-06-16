/*
 * Review-response benchmark: adversarial performance curves for R5.Q3.
 *
 * This benchmark complements the correctness matrix.  It measures how HKV's
 * 128B value-returning embedding/KV-cache workload behaves under concentrated
 * bucket pressure, Zipf-like hot keys, and legal overlapping R/U/I streams.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "merlin/utils.cuh"
#include "merlin_hashtable.cuh"

namespace {

constexpr size_t kDim = 32;
constexpr size_t kBucketSize = 128;

using K = uint64_t;
using V = float;
using S = uint64_t;
using Table =
    nv::merlin::HashTable<K, V, S, nv::merlin::EvictStrategy::kCustomized>;

struct Config {
  std::string mode = "quick";
  std::string scenario = "all";
  size_t capacity = 1UL << 20;
  size_t batch = 1UL << 16;
  size_t adversarial_batch = 1UL << 15;
  int streams = 6;
  int warmup = 1;
  int runs = 3;
  double alpha = 1.25;
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
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  void copy_from_host(const std::vector<T>& host,
                      cudaStream_t stream = 0) const {
    if (!host.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(ptr_, host.data(), host.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    }
  }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

struct CudaTimer {
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  double stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return static_cast<double>(ms) * 1e-3;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

std::vector<K> continuous_keys(size_t n, K start = 1) {
  std::vector<K> keys(n);
  std::iota(keys.begin(), keys.end(), start);
  return keys;
}

std::vector<S> scores_with_base(size_t n, S base) {
  std::vector<S> scores(n);
  for (size_t i = 0; i < n; i++) {
    scores[i] = base + static_cast<S>(i);
  }
  return scores;
}

std::vector<V> sentinel_values(size_t n, V sentinel) {
  return std::vector<V>(n * kDim, sentinel);
}

std::vector<K> zipf_like_keys(size_t n, size_t hot_keys, double alpha,
                              K start) {
  std::vector<double> weights(hot_keys);
  for (size_t i = 0; i < hot_keys; i++) {
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
  }
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  std::mt19937_64 rng(20260611);

  std::vector<K> keys(n);
  for (size_t i = 0; i < n; i++) {
    keys[i] = start + static_cast<K>(dist(rng));
  }
  return keys;
}

size_t throughput_bucket(K key, size_t capacity) {
  const uint64_t hash = nv::merlin::Murmur3HashHost(key);
  const size_t global_idx = hash & (capacity - 1);
  return global_idx / kBucketSize;
}

std::vector<K> keys_in_bucket_set(size_t count, size_t capacity,
                                  size_t target_buckets, K start) {
  const size_t total_buckets = capacity / kBucketSize;
  target_buckets = std::max<size_t>(1, std::min(target_buckets, total_buckets));

  std::vector<K> keys;
  keys.reserve(count);
  for (K candidate = start; keys.size() < count; candidate++) {
    if (throughput_bucket(candidate, capacity) < target_buckets) {
      keys.push_back(candidate);
    }
  }
  return keys;
}

void create_table(Table& table, size_t capacity, bool api_lock = true) {
  nv::merlin::HashTableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_bucket_size = kBucketSize;
  options.max_load_factor = 1.0f;
  options.dim = kDim;
  options.table_mode = nv::merlin::TableMode::kThroughput;
  options.max_hbm_for_vectors = capacity * kDim * sizeof(V);
  options.api_lock = api_lock;
  table.init(options);
}

void insert_or_assign(Table& table, const std::vector<K>& keys,
                      const std::vector<V>& values,
                      const std::vector<S>& scores, bool unique_key,
                      bool ignore_evict_strategy = false,
                      cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(values.size());
  DeviceBuffer<S> d_scores(scores.size());
  d_keys.copy_from_host(keys, stream);
  d_values.copy_from_host(values, stream);
  d_scores.copy_from_host(scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  table.insert_or_assign(keys.size(), d_keys.get(), d_values.get(),
                         d_scores.get(), stream, unique_key,
                         ignore_evict_strategy);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

double timed_insert_or_assign(Table& table, const std::vector<K>& keys,
                              const std::vector<V>& values,
                              const std::vector<S>& scores, bool unique_key,
                              bool ignore_evict_strategy = false,
                              cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(values.size());
  DeviceBuffer<S> d_scores(scores.size());
  d_keys.copy_from_host(keys, stream);
  d_values.copy_from_host(values, stream);
  d_scores.copy_from_host(scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CudaTimer timer;
  timer.start(stream);
  table.insert_or_assign(keys.size(), d_keys.get(), d_values.get(),
                         d_scores.get(), stream, unique_key,
                         ignore_evict_strategy);
  return timer.stop(stream);
}

double timed_find(Table& table, const std::vector<K>& keys,
                  cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(keys.size() * kDim);
  DeviceBuffer<bool> d_founds(keys.size());
  d_keys.copy_from_host(keys, stream);
  CUDA_CHECK(
      cudaMemsetAsync(d_founds.get(), 0, keys.size() * sizeof(bool), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CudaTimer timer;
  timer.start(stream);
  table.find(keys.size(), d_keys.get(), d_values.get(), d_founds.get(), nullptr,
             stream);
  return timer.stop(stream);
}

void assign_values(Table& table, const std::vector<K>& keys,
                   const std::vector<V>& values, const std::vector<S>& scores,
                   cudaStream_t stream = 0) {
  DeviceBuffer<K> d_keys(keys.size());
  DeviceBuffer<V> d_values(values.size());
  DeviceBuffer<S> d_scores(scores.size());
  d_keys.copy_from_host(keys, stream);
  d_values.copy_from_host(values, stream);
  d_scores.copy_from_host(scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  table.assign(keys.size(), d_keys.get(), d_values.get(), d_scores.get(),
               stream, true);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

double bkvs(size_t ops, double seconds) {
  return static_cast<double>(ops) / seconds / 1.0e9;
}

void print_header() {
  std::cout << "scenario,operation,capacity,batch,hot_buckets,streams,run,"
               "seconds,throughput_bkvs,final_size,notes\n";
}

void print_row(const std::string& scenario, const std::string& operation,
               size_t capacity, size_t batch, size_t hot_buckets, int streams,
               int run, double seconds, uint64_t final_size,
               const std::string& notes) {
  std::cout << scenario << ',' << operation << ',' << capacity << ',' << batch
            << ',' << hot_buckets << ',' << streams << ',' << run << ','
            << std::fixed << std::setprecision(9) << seconds << ','
            << std::setprecision(6) << bkvs(batch, seconds) << ',' << final_size
            << ',' << notes << '\n';
}

bool want(const Config& cfg, const std::string& scenario) {
  return cfg.scenario == "all" || cfg.scenario == scenario;
}

void run_insert_scenario(const Config& cfg, const std::string& scenario,
                         const std::vector<K>& keys, bool unique_key,
                         size_t hot_buckets, const std::string& notes) {
  std::vector<V> values = sentinel_values(keys.size(), 1.0f);
  std::vector<S> scores = scores_with_base(keys.size(), 1000);

  for (int run = -cfg.warmup; run < cfg.runs; run++) {
    Table table;
    create_table(table, cfg.capacity, true);
    double seconds =
        timed_insert_or_assign(table, keys, values, scores, unique_key, false);
    if (run >= 0) {
      print_row(scenario, "insert_or_assign", cfg.capacity, keys.size(),
                hot_buckets, cfg.streams, run + 1, seconds, table.size(0),
                notes);
    }
  }
}

void run_find_scenario(const Config& cfg, const std::string& scenario,
                       const std::vector<K>& keys, size_t hot_buckets,
                       const std::string& notes) {
  const size_t fill =
      std::min(cfg.capacity / 2, std::max(cfg.batch, keys.size()));
  std::vector<K> fill_keys = continuous_keys(fill, 1);
  std::vector<V> fill_values = sentinel_values(fill, 2.0f);
  std::vector<S> fill_scores = scores_with_base(fill, 1000);

  for (int run = -cfg.warmup; run < cfg.runs; run++) {
    Table table;
    create_table(table, cfg.capacity, true);
    insert_or_assign(table, fill_keys, fill_values, fill_scores, true, true);
    double seconds = timed_find(table, keys);
    if (run >= 0) {
      print_row(scenario, "find", cfg.capacity, keys.size(), hot_buckets,
                cfg.streams, run + 1, seconds, table.size(0), notes);
    }
  }
}

class Barrier {
 public:
  explicit Barrier(int target) : target_(target) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    arrived_++;
    if (arrived_ == target_) {
      released_ = true;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [this] { return released_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int target_;
  int arrived_ = 0;
  bool released_ = false;
};

double run_rui_once(const Config& cfg, bool concurrent, uint64_t* attempted,
                    uint64_t* final_size) {
  const int workers = std::max(3, cfg.streams);
  const size_t initial = std::min(cfg.capacity / 2, cfg.batch * 2);
  const size_t op_batch = std::max<size_t>(1024, cfg.batch / workers);

  Table table;
  create_table(table, cfg.capacity, true);
  insert_or_assign(table, continuous_keys(initial, 1),
                   sentinel_values(initial, 1.0f),
                   scores_with_base(initial, 1000), true, true);

  std::atomic<uint64_t> total_ops{0};
  std::atomic<bool> failed{false};
  Barrier barrier(workers);

  auto worker = [&](int worker_id) {
    try {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      if (concurrent) {
        barrier.wait();
      }
      const size_t offset = static_cast<size_t>(worker_id) * op_batch;
      std::vector<K> keys(op_batch);
      if (worker_id % 3 == 2) {
        for (size_t i = 0; i < op_batch / 2; i++) {
          keys[i] = 1 + static_cast<K>((i + offset) % initial);
        }
        for (size_t i = op_batch / 2; i < op_batch; i++) {
          keys[i] = 9000000 + static_cast<K>(offset + i);
        }
        insert_or_assign(
            table, keys,
            sentinel_values(op_batch, static_cast<V>(30 + worker_id % 10)),
            scores_with_base(op_batch, 9000 + offset), true, false, stream);
      } else {
        for (size_t i = 0; i < op_batch; i++) {
          keys[i] = 1 + static_cast<K>((i + offset) % initial);
        }
        if (worker_id % 3 == 0) {
          (void)timed_find(table, keys, stream);
        } else {
          assign_values(
              table, keys,
              sentinel_values(op_batch, static_cast<V>(20 + worker_id % 10)),
              scores_with_base(op_batch, 5000 + offset), stream);
        }
      }
      total_ops.fetch_add(op_batch);
      CUDA_CHECK(cudaStreamDestroy(stream));
    } catch (...) {
      failed = true;
    }
  };

  const auto start = std::chrono::steady_clock::now();
  if (concurrent) {
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int worker_id = 0; worker_id < workers; worker_id++) {
      threads.emplace_back(worker, worker_id);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  } else {
    for (int worker_id = 0; worker_id < workers; worker_id++) {
      worker(worker_id);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto end = std::chrono::steady_clock::now();

  if (failed.load()) {
    std::cerr << "R/U/I worker failed\n";
  }
  *attempted = total_ops.load();
  *final_size = table.size(0);
  return std::chrono::duration<double>(end - start).count();
}

void run_rui_scenario(const Config& cfg) {
  for (int run = -cfg.warmup; run < cfg.runs; run++) {
    uint64_t ops = 0;
    uint64_t size = 0;
    double seconds = run_rui_once(cfg, false, &ops, &size);
    if (run >= 0) {
      print_row("rui_serial", "mixed_find_assign_insert", cfg.capacity, ops, 0,
                cfg.streams, run + 1, seconds, size,
                "same roles executed sequentially");
    }
  }
  for (int run = -cfg.warmup; run < cfg.runs; run++) {
    uint64_t ops = 0;
    uint64_t size = 0;
    double seconds = run_rui_once(cfg, true, &ops, &size);
    if (run >= 0) {
      print_row("rui_overlap", "mixed_find_assign_insert", cfg.capacity, ops, 0,
                cfg.streams, run + 1, seconds, size,
                "api_lock=true legal R/U/I overlap");
    }
  }
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    auto value = [&](const std::string& prefix) {
      return arg.substr(prefix.size());
    };
    if (arg == "--mode=quick") {
      cfg.mode = "quick";
      cfg.capacity = 1UL << 20;
      cfg.batch = 1UL << 16;
      cfg.adversarial_batch = 1UL << 15;
      cfg.warmup = 1;
      cfg.runs = 3;
    } else if (arg == "--mode=full") {
      cfg.mode = "full";
      cfg.capacity = 16UL * 1024 * 1024;
      cfg.batch = 1024UL * 1024;
      cfg.adversarial_batch = 256UL * 1024;
      cfg.warmup = 2;
      cfg.runs = 5;
    } else if (arg.rfind("--scenario=", 0) == 0) {
      cfg.scenario = value("--scenario=");
    } else if (arg.rfind("--capacity=", 0) == 0) {
      cfg.capacity = std::strtoull(value("--capacity=").c_str(), nullptr, 10);
    } else if (arg.rfind("--batch=", 0) == 0) {
      cfg.batch = std::strtoull(value("--batch=").c_str(), nullptr, 10);
    } else if (arg.rfind("--adversarial_batch=", 0) == 0) {
      cfg.adversarial_batch =
          std::strtoull(value("--adversarial_batch=").c_str(), nullptr, 10);
    } else if (arg.rfind("--streams=", 0) == 0) {
      cfg.streams = std::atoi(value("--streams=").c_str());
    } else if (arg.rfind("--warmup=", 0) == 0) {
      cfg.warmup = std::atoi(value("--warmup=").c_str());
    } else if (arg.rfind("--runs=", 0) == 0) {
      cfg.runs = std::atoi(value("--runs=").c_str());
    } else if (arg.rfind("--alpha=", 0) == 0) {
      cfg.alpha = std::atof(value("--alpha=").c_str());
    } else if (arg == "--help" || arg == "-h") {
      std::cerr
          << "Usage: hkv_adversarial_performance_benchmark "
             "[--mode=quick|full]\n"
          << "       [--scenario=all|uniform|bucket_skew|single_bucket|zipf|"
             "rui]\n"
          << "       [--capacity=N] [--batch=N] [--adversarial_batch=N]\n"
          << "       [--streams=N] [--warmup=N] [--runs=N] [--alpha=A]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  if ((cfg.capacity & (cfg.capacity - 1)) != 0) {
    std::cerr << "--capacity must be a power of two\n";
    std::exit(2);
  }
  if (cfg.capacity < kBucketSize * 2 || cfg.batch == 0 ||
      cfg.adversarial_batch == 0) {
    std::cerr << "invalid capacity or batch\n";
    std::exit(2);
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg = parse_args(argc, argv);
  CUDA_CHECK(cudaFree(0));
  print_header();

  if (want(cfg, "uniform")) {
    auto keys = continuous_keys(cfg.batch, 1);
    run_insert_scenario(cfg, "uniform", keys, true, cfg.capacity / kBucketSize,
                        "uniform Murmur3 bucket distribution");
    run_find_scenario(cfg, "uniform", keys, cfg.capacity / kBucketSize,
                      "uniform positive 128B value-returning find");
  }

  if (want(cfg, "bucket_skew")) {
    const size_t total_buckets = cfg.capacity / kBucketSize;
    const std::vector<size_t> divisors = {4, 16, 64, 256};
    for (size_t divisor : divisors) {
      const size_t hot_buckets = std::max<size_t>(1, total_buckets / divisor);
      auto keys =
          keys_in_bucket_set(cfg.batch, cfg.capacity, hot_buckets,
                             10000000 + static_cast<K>(divisor) * 1000000ULL);
      std::ostringstream scenario;
      scenario << "bucket_skew_1over" << divisor;
      std::ostringstream notes;
      notes << "keys constrained to first 1/" << divisor << " buckets";
      run_insert_scenario(cfg, scenario.str(), keys, true, hot_buckets,
                          notes.str());
    }
  }

  if (want(cfg, "single_bucket")) {
    const size_t single_bucket_batch =
        std::min(cfg.adversarial_batch, kBucketSize * 4);
    auto keys =
        keys_in_bucket_set(single_bucket_batch, cfg.capacity, 1, 20000000);
    run_insert_scenario(
        cfg, "single_bucket_pressure", keys, true, 1,
        "small extreme point: all keys constrained to one bucket");
  }

  if (want(cfg, "zipf")) {
    const size_t hot_keys = std::max<size_t>(1, cfg.batch / 16);
    auto keys = zipf_like_keys(cfg.batch, hot_keys, cfg.alpha, 30000000);
    run_insert_scenario(cfg, "zipf_hotset", keys, false, hot_keys,
                        "Zipf-like repeated keys; unique_key=false");
  }

  if (want(cfg, "rui")) {
    run_rui_scenario(cfg);
  }

  return 0;
}
