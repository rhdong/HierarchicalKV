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
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "benchmark_util.cuh"
#include "merlin_hashtable.cuh"

using K = uint64_t;
using S = uint64_t;
using V = float;
using namespace nv::merlin;
using namespace benchmark;

static constexpr size_t DIM = 32;
static constexpr size_t INIT_CAPACITY = 128UL * 1024 * 1024;
static constexpr size_t HBM_GB = 16;
static constexpr size_t BATCH_SIZE = 1024 * 1024UL;
static constexpr int WARMUP = 3;
static constexpr int RUNS = 5;

using HkvTable = HashTable<K, V, S, EvictStrategy::kLru, Sm80>;

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

  auto table = std::make_shared<HkvTable>();
  table->init(options);
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  prepopulate(table, target_n, stream);
  return table;
}

static void prepare_queries(K* h_keys, float target_lf) {
  const size_t target_n = static_cast<size_t>(INIT_CAPACITY * target_lf);
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<K> dist(0, target_n - 1);
  for (size_t i = 0; i < BATCH_SIZE; i++) {
    h_keys[i] = dist(rng);
  }
}

static void run_lookup_modes(float target_lf, cudaStream_t stream) {
  std::cerr << "HKV lookup modes LF=" << std::fixed << std::setprecision(2)
            << target_lf << std::endl;

  auto table = build_table(target_lf, stream);

  K* h_keys;
  K* d_keys;
  V* d_vectors;
  bool* d_found;
  CUDA_CHECK(cudaMallocHost(&h_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_keys, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_vectors, BATCH_SIZE * sizeof(V) * DIM));
  CUDA_CHECK(cudaMalloc(&d_found, BATCH_SIZE * sizeof(bool)));

  prepare_queries(h_keys, target_lf);
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys, BATCH_SIZE * sizeof(K),
                        cudaMemcpyHostToDevice));

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

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
}

int main(int argc, char** argv) {
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
  std::cerr << "GPU: " << props.name << std::endl;
  std::cerr << "Review P0: HKV lookup-mode audit" << std::endl;
  std::cerr << "Config B: dim=" << DIM << ", capacity=" << INIT_CAPACITY
            << ", HBM=" << HBM_GB << "GB, kLru" << std::endl;

  std::vector<float> load_factors;
  if (argc > 1) {
    for (int i = 1; i < argc; i++) {
      load_factors.push_back(std::stof(argv[i]));
    }
  } else {
    load_factors = {0.25f, 0.50f, 0.75f, 1.00f};
  }

  std::cout << "library,mode,operation,load_factor,run,throughput_bkvs"
            << std::endl;

  for (float lf : load_factors) {
    run_lookup_modes(lf, 0);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
