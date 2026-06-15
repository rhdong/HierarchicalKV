/*
 * Review-response P0: HKV lookup-mode audit.
 *
 * Measures key-only membership lookup (`contains`) against value-returning
 * lookup (`find`) under the same table and query setup as the E8 baseline.
 *
 * Output: CSV
 *   library,mode,operation,load_factor,run,throughput_bkvs
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using namespace nv::merlin;
using namespace benchmark;

#ifndef HKV_LOOKUP_DIM
#define HKV_LOOKUP_DIM 32
#endif
#ifndef HKV_LOOKUP_CAPACITY
#define HKV_LOOKUP_CAPACITY 134217728
#endif
#ifndef HKV_LOOKUP_BATCH_SIZE
#define HKV_LOOKUP_BATCH_SIZE 1048576
#endif
#ifndef HKV_LOOKUP_WARMUP
#define HKV_LOOKUP_WARMUP 3
#endif
#ifndef HKV_LOOKUP_RUNS
#define HKV_LOOKUP_RUNS 5
#endif
#ifndef HKV_LOOKUP_HBM_GB
#define HKV_LOOKUP_HBM_GB 16
#endif
#ifndef HKV_LOOKUP_BLOCK_SIZE
#define HKV_LOOKUP_BLOCK_SIZE 128
#endif
#ifndef HKV_LOOKUP_API_LOCK
#define HKV_LOOKUP_API_LOCK 1
#endif

static constexpr size_t DIM = HKV_LOOKUP_DIM;
static constexpr size_t INIT_CAPACITY = HKV_LOOKUP_CAPACITY;
static constexpr size_t HBM_GB = HKV_LOOKUP_HBM_GB;
static constexpr size_t BATCH_SIZE = HKV_LOOKUP_BATCH_SIZE;
static constexpr int WARMUP = HKV_LOOKUP_WARMUP;
static constexpr int RUNS = HKV_LOOKUP_RUNS;
static constexpr int BLOCK_SIZE = HKV_LOOKUP_BLOCK_SIZE;
static constexpr bool API_LOCK = (HKV_LOOKUP_API_LOCK != 0);

using HkvTable = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

__global__ void touch_ptr_values_kernel(V** ptrs, const bool* founds,
                                        unsigned long long* errors, size_t dim,
                                        size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  float acc = 0.0f;
  for (size_t d = 0; d < dim; d++) {
    acc += ptrs[idx][d];
  }
  if (acc == -1.0f) {
    atomicAdd(errors, 1ULL);
  }
}

__global__ void gather_ptr_values_kernel(V** ptrs, const bool* founds, V* out,
                                         unsigned long long* errors, size_t dim,
                                         size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * dim;
  if (tid >= total) return;
  const size_t idx = tid / dim;
  const size_t d = tid - idx * dim;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  out[tid] = ptrs[idx][d];
}

__global__ void gather_ptr_values_by_key_kernel(V** ptrs, const bool* founds,
                                                V* out,
                                                unsigned long long* errors,
                                                size_t dim, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  V* src = ptrs[idx];
  V* dst = out + idx * dim;
  for (size_t d = 0; d < dim; d++) {
    dst[d] = src[d];
  }
}

__global__ void gather_ptr_values_warp_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t dim, size_t n) {
  constexpr int WARP_SIZE = 32;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  if (warp_id >= n) return;
  if (!founds[warp_id] || ptrs[warp_id] == nullptr) {
    if (lane == 0) atomicAdd(errors, 1ULL);
    return;
  }
  V* src = ptrs[warp_id];
  V* dst = out + warp_id * dim;
  for (size_t d = lane; d < dim; d += WARP_SIZE) {
    dst[d] = src[d];
  }
}

__global__ void gather_ptr_values_vec2_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float2* src = reinterpret_cast<const float2*>(ptrs[idx]);
  float2* dst = reinterpret_cast<float2*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_vec4_kernel(V** ptrs, const bool* founds,
                                              V* out,
                                              unsigned long long* errors,
                                              size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (!founds[idx] || ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float4* src = reinterpret_cast<const float4*>(ptrs[idx]);
  float4* dst = reinterpret_cast<float4*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_nofound_kernel(V** ptrs, V* out,
                                                 unsigned long long* errors,
                                                 size_t dim, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * dim;
  if (tid >= total) return;
  const size_t idx = tid / dim;
  const size_t d = tid - idx * dim;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  out[tid] = ptrs[idx][d];
}

__global__ void gather_ptr_values_vec2_nofound_kernel(
    V** ptrs, V* out, unsigned long long* errors, size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float2* src = reinterpret_cast<const float2*>(ptrs[idx]);
  float2* dst = reinterpret_cast<float2*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

__global__ void gather_ptr_values_vec4_nofound_kernel(
    V** ptrs, V* out, unsigned long long* errors, size_t chunks, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = n * chunks;
  if (tid >= total) return;
  const size_t idx = tid / chunks;
  const size_t chunk = tid - idx * chunks;
  if (ptrs[idx] == nullptr) {
    atomicAdd(errors, 1ULL);
    return;
  }
  const float4* src = reinterpret_cast<const float4*>(ptrs[idx]);
  float4* dst = reinterpret_cast<float4*>(out);
  dst[idx * chunks + chunk] = src[chunk];
}

static K prepopulate(std::shared_ptr<HkvTable>& table, size_t target_count,
                     cudaStream_t stream) {
  K* h_keys;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));

  K* d_keys;
  V* d_vectors;
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMemset(d_vectors, 1, BATCH_SIZE * sizeof(V) * DIM));

  K start = 0;
  while (start < target_count) {
    size_t cur = std::min(BATCH_SIZE, target_count - start);
    for (size_t i = 0; i < cur; i++) h_keys[i] = start + i;
    CUDA_CHECK(
        cudaMemcpy(d_keys, h_keys, cur * sizeof(K), cudaMemcpyHostToDevice));
    table->insert_or_assign(cur, d_keys, d_vectors, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    start += cur;
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  return start;
}

static std::shared_ptr<HkvTable> build_table(float target_lf,
                                             cudaStream_t stream) {
  HashTableOptions options;
  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = INIT_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(HBM_GB);
  options.block_size = BLOCK_SIZE;
  options.api_lock = API_LOCK;

  auto table = std::make_shared<HkvTable>();
  table->init(options);
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  prepopulate(table, target_n, stream);
  return table;
}

static void prepare_queries(K* h_keys, float target_lf,
                            const std::string& query_pattern) {
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  if (query_pattern == "sequential") {
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_keys[i] = static_cast<K>(i % target_n);
    }
  } else {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<K> dist(0, target_n - 1);
    for (size_t i = 0; i < BATCH_SIZE; i++) {
      h_keys[i] = dist(rng);
    }
  }
}

static void verify_all_found(const bool* d_found, const std::string& label) {
  std::vector<uint8_t> h_found(BATCH_SIZE);
  CUDA_CHECK(cudaMemcpy(h_found.data(), d_found, BATCH_SIZE * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  size_t found_count = 0;
  for (uint8_t found : h_found) {
    found_count += (found != 0);
  }
  std::cerr << label << " found=" << found_count << "/" << BATCH_SIZE
            << std::endl;
  if (found_count != BATCH_SIZE) {
    throw std::runtime_error(label + " failed successful-query check");
  }
}

static void run_lookup_modes(float target_lf, const std::string& mode,
                             const std::string& query_pattern,
                             cudaStream_t stream) {
  std::cerr << "HKV lookup modes LF=" << std::fixed << std::setprecision(2)
            << target_lf << std::endl;

  auto table = build_table(target_lf, stream);

  K* h_keys;
  K* d_keys;
  V* d_vectors;
  V** d_value_ptrs;
  bool* d_found;
  unsigned long long* d_errors;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_value_ptrs, BATCH_SIZE * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_errors, sizeof(unsigned long long)));

  prepare_queries(h_keys, target_lf, query_pattern);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));

  if (mode == "both" || mode == "all" || mode == "academic" ||
      mode == "key_only") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->contains(BATCH_SIZE, d_keys, d_found, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,key_only,contains," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    verify_all_found(d_found, "HKV key_only contains");
  }

  if (mode == "all" || mode == "academic" || mode == "ptr_only" ||
      mode == "ptr_touch" || mode == "ptr_gather" ||
      mode == "ptr_only_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs,
                  mode == "ptr_only_nofound" ? nullptr : d_found, nullptr,
                  stream, true);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,"
                  << (mode == "ptr_only_nofound" ? "academic_ptr_only_nofound"
                                                 : "academic_ptr_only")
                  << ","
                  << (mode == "ptr_only_nofound" ? "find_ptr_nofound"
                                                 : "find_ptr")
                  << "," << std::fixed << std::setprecision(2) << target_lf
                  << "," << (run - WARMUP + 1) << "," << std::setprecision(6)
                  << tp << std::endl;
      }
    }
    if (mode != "ptr_only_nofound") {
      verify_all_found(d_found, "HKV academic ptr-only find");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "ptr_touch") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      touch_ptr_values_kernel<<<(BATCH_SIZE + 255) / 256, 256, 0, stream>>>(
          d_value_ptrs, d_found, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_touch,find_ptr_touch," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-touch failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_kernel<<<(BATCH_SIZE * DIM + 255) / 256, 256, 0,
                                 stream>>>(d_value_ptrs, d_found, d_vectors,
                                           d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather,find_ptr_gather," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather_key") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_by_key_kernel<<<(BATCH_SIZE + 255) / 256, 256, 0,
                                        stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_key,find_ptr_gather_key,"
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-key failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "gather_sweep" ||
      mode == "ptr_gather_warp") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_warp_kernel<<<(BATCH_SIZE * 32 + 127) / 128, 128, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_warp,find_ptr_gather_warp,"
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-warp failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "gather_sweep" ||
       mode == "ptr_gather_vec2") &&
      DIM % 2 == 0) {
    const size_t chunks = DIM / 2;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_vec2_kernel<<<(BATCH_SIZE * chunks + 255) / 256, 256, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_vec2,find_ptr_gather_vec2,"
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec2 failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "gather_sweep" ||
       mode == "ptr_gather_vec4") &&
      DIM % 4 == 0) {
    const size_t chunks = DIM / 4;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, d_found, nullptr, stream,
                  true);
      gather_ptr_values_vec4_kernel<<<(BATCH_SIZE * chunks + 255) / 256, 256, 0,
                                      stream>>>(
          d_value_ptrs, d_found, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_vec4,find_ptr_gather_vec4,"
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec4 failed");
    }
  }

  if (mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
      mode == "ptr_gather_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_nofound_kernel<<<(BATCH_SIZE * DIM + 255) / 256, 256, 0,
                                         stream>>>(d_value_ptrs, d_vectors,
                                                   d_errors, DIM, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_nofound,"
                  << "find_ptr_gather_nofound," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-nofound failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
       mode == "ptr_gather_vec2_nofound") &&
      DIM % 2 == 0) {
    const size_t chunks = DIM / 2;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_vec2_nofound_kernel<<<(BATCH_SIZE * chunks + 255) / 256,
                                              256, 0, stream>>>(
          d_value_ptrs, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_vec2_nofound,"
                  << "find_ptr_gather_vec2_nofound," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec2-nofound failed");
    }
  }

  if ((mode == "all" || mode == "academic" || mode == "nofound_sweep" ||
       mode == "ptr_gather_vec4_nofound") &&
      DIM % 4 == 0) {
    const size_t chunks = DIM / 4;
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_value_ptrs, 0, BATCH_SIZE * sizeof(V*), stream));
      CUDA_CHECK(
          cudaMemsetAsync(d_errors, 0, sizeof(unsigned long long), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_value_ptrs, nullptr, nullptr, stream,
                  true);
      gather_ptr_values_vec4_nofound_kernel<<<(BATCH_SIZE * chunks + 255) / 256,
                                              256, 0, stream>>>(
          d_value_ptrs, d_vectors, d_errors, chunks, BATCH_SIZE);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_ptr_gather_vec4_nofound,"
                  << "find_ptr_gather_vec4_nofound," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    unsigned long long h_errors = 0;
    CUDA_CHECK(cudaMemcpy(&h_errors, d_errors, sizeof(h_errors),
                          cudaMemcpyDeviceToHost));
    if (h_errors != 0) {
      throw std::runtime_error("HKV academic ptr-gather-vec4-nofound failed");
    }
  }

  if (mode == "both" || mode == "all" || mode == "value_returning") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find(BATCH_SIZE, d_keys, d_vectors, d_found, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,value_returning,find," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    verify_all_found(d_found, "HKV value_returning find");
  }

  if (mode == "all" || mode == "academic" || mode == "readonly_fast_sweep" ||
      mode == "readonly_fast") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      CUDA_CHECK(
          cudaMemsetAsync(d_found, 0, BATCH_SIZE * sizeof(bool), stream));
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find_readonly_fast(BATCH_SIZE, d_keys, d_vectors, d_found, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_readonly_fast,find_readonly_fast,"
                  << std::fixed << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
    verify_all_found(d_found, "HKV readonly_fast find");
  }

  if (mode == "all" || mode == "academic" || mode == "readonly_fast_sweep" ||
      mode == "readonly_fast_nofound") {
    for (int run = 0; run < WARMUP + RUNS; run++) {
      auto timer = benchmark::KernelTimer<double>();
      timer.start();
      table->find_readonly_fast(BATCH_SIZE, d_keys, d_vectors, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      timer.end();

      if (run >= WARMUP) {
        double tp = BATCH_SIZE / timer.getResult() / (1024.0 * 1024.0 * 1024.0);
        std::cout << "HKV,academic_readonly_fast_nofound,"
                  << "find_readonly_fast_nofound," << std::fixed
                  << std::setprecision(2) << target_lf << ","
                  << (run - WARMUP + 1) << "," << std::setprecision(6) << tp
                  << std::endl;
      }
    }
  }

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_value_ptrs));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_errors));
}

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review P0: HKV lookup-mode audit" << std::endl;
  std::cerr << "Config: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", batch=" << BATCH_SIZE << ", warmup=" << WARMUP
            << ", runs=" << RUNS << ", block=" << BLOCK_SIZE
            << ", HBM=" << HBM_GB
            << "GB, api_lock=" << (API_LOCK ? "on" : "off") << ", kLru"
            << std::endl;

  std::string mode = "both";
  std::string query_pattern = "random";
  std::vector<float> load_factors;
  for (int i = 1; i < argc; i++) {
    std::string arg(argv[i]);
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg.rfind("--mode=", 0) == 0) {
      mode = arg.substr(7);
    } else if (arg == "--query" && i + 1 < argc) {
      query_pattern = argv[++i];
    } else if (arg.rfind("--query=", 0) == 0) {
      query_pattern = arg.substr(8);
    } else {
      load_factors.push_back(std::stof(arg));
    }
  }

  if (mode != "both" && mode != "all" && mode != "key_only" &&
      mode != "value_returning" && mode != "academic" && mode != "ptr_only" &&
      mode != "ptr_touch" && mode != "ptr_gather" && mode != "gather_sweep" &&
      mode != "ptr_gather_key" && mode != "ptr_gather_warp" &&
      mode != "ptr_gather_vec2" && mode != "ptr_gather_vec4" &&
      mode != "nofound_sweep" && mode != "ptr_only_nofound" &&
      mode != "ptr_gather_nofound" && mode != "ptr_gather_vec2_nofound" &&
      mode != "ptr_gather_vec4_nofound" && mode != "readonly_fast_sweep" &&
      mode != "readonly_fast" && mode != "readonly_fast_nofound") {
    std::cerr << "Unknown mode: " << mode
              << " (expected both, all, key_only, value_returning, academic, "
                 "ptr_only, ptr_touch, ptr_gather, gather_sweep, "
                 "ptr_gather_key, ptr_gather_warp, ptr_gather_vec2, "
                 "ptr_gather_vec4, nofound_sweep, ptr_only_nofound, "
                 "ptr_gather_nofound, ptr_gather_vec2_nofound, "
                 "ptr_gather_vec4_nofound, readonly_fast_sweep, "
                 "readonly_fast, readonly_fast_nofound)"
              << std::endl;
    return 1;
  }
  if (query_pattern != "random" && query_pattern != "sequential") {
    std::cerr << "Unknown query pattern: " << query_pattern
              << " (expected random or sequential)" << std::endl;
    return 1;
  }

  if (load_factors.empty()) {
    load_factors = {0.25f, 0.50f, 0.75f, 1.00f};
  }

  std::cout << "library,mode,operation,load_factor,run,throughput_bkvs"
            << std::endl;

  for (float lf : load_factors) {
    run_lookup_modes(lf, mode, query_pattern, 0);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
