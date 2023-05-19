#include <gtest/gtest.h>
#include <stdio.h>
#include <array>
#include <map>
#include "cuda_runtime.h"
#include "merlin/types.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_localfile.hpp"
#include "test_util.cuh"

constexpr size_t dim = 64;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using Table = nv::merlin::HashTable<i64, f32, u64>;
using TableOptions = nv::merlin::HashTableOptions;

template <class K, class M>
__forceinline__ __device__ bool export_if_pred(const K& key, M& meta,
                                               const K& pattern,
                                               const M& threshold) {
  return meta > threshold;
}

template <class K, class M>
__device__ Table::Pred ExportIfPred = export_if_pred<K, M>;

void test_export_with_condition() {
  TableOptions opt;

  // table setting
  const size_t init_capacity = 32;

  // numeric setting
  const size_t U = 2llu << 18;
  const size_t M = (U >> 1);
  const size_t N = (U >> 1) + 17;  // Add a prime to test the non-aligned case.

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // step1
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> buffer;
  buffer.Reserve(M, dim, stream);
  buffer.ToRange(0, 1, stream);
  buffer.SetMeta((u64)1, stream);

  u64* h_metas = buffer.metas_ptr(false);
  for (size_t i = 0; i < M; i++) {
    h_metas[i] = static_cast<u64>(i);
  }
  buffer.SyncData(true, stream);
  table->insert_or_assign(M, buffer.keys_ptr(), buffer.values_ptr(),
                          buffer.metas_ptr(), stream);

  i64 pattern = 0;
  u64 threshold = M / 2;
  size_t* d_counter = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_counter, sizeof(size_t), stream));
  CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(size_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  table->export_batch_if(ExportIfPred<i64, u64>, pattern, threshold,
                         table->capacity(), 0, d_counter, buffer.keys_ptr(),
                         buffer.values_ptr(), buffer.metas_ptr(), stream);

  size_t h_counter = 0;
  buffer.Free(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpyAsync(&h_counter, d_counter, sizeof(size_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_counter, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  printf("----> check h_counter: %llu\n", h_counter);
}

TEST(ExportWithCondition, test_export_with_condition) {
  test_export_with_condition();
}