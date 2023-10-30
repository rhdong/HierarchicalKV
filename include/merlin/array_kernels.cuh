/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:///www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime.h"
#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/scan.h"
#include "types.cuh"
#include "utils.cuh"

namespace nv {
namespace merlin {

template <typename K>
__global__ void keys_not_empty(const K* keys, bool* masks, size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    masks[tid] = keys[tid] != EMPTY_KEY;
  }
}

template <typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_cell_count(const bool* masks, Tidx* offsets, size_t n,
                               size_t* n_existed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();
  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  int g_ones = __popc((int)vote);
  if (rank == 0 && tid < n) {
    offsets[tid / TILE_SIZE] = static_cast<Tidx>(g_ones);
    atomicAdd(static_cast<uint64_t*>(n_existed), static_cast<uint64_t>(g_ones));
  }
}

template <typename K, typename V, typename S, typename Tidx, int TILE_SIZE = 8>
__global__ void gpu_select_kvm_kernel(const bool* masks, size_t n,
                                      const Tidx* offsets, K* __restrict keys,
                                      V* __restrict values,
                                      S* __restrict scores, const size_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();

  bool is_existed = false;
  if (tid < n) {
    if (masks[tid]) {
      is_existed = true;
    }
  }
  unsigned int vote = g.ballot(is_existed);
  unsigned int r_vote = __brev(vote) >> (32 - TILE_SIZE);
  K empty_key = (K)EMPTY_KEY;

  if (tid < n) {
    r_vote = r_vote >> (TILE_SIZE - rank - 1);
    if (masks[tid]) {
      int prefix_n = __popc(r_vote) - 1;
      Tidx bias = offsets[tid / TILE_SIZE] + static_cast<Tidx>(prefix_n);

      if (bias == tid) return;

      K target_key = 0;
      AtomicKey<K>* atomic_key = reinterpret_cast<AtomicKey<K>*>(keys) + bias;
      while (target_key != empty_key) {
        target_key = empty_key;
        atomic_key->compare_exchange_weak(target_key, keys[tid],
                                          cuda::std::memory_order_relaxed,
                                          cuda::std::memory_order_relaxed);
      }
      if (scores) scores[bias] = scores[tid];
      for (size_t j = 0; j < dim; j++) {
        values[dim * bias + j] = values[dim * tid + j];
      }
      atomic_key = reinterpret_cast<AtomicKey<K>*>(keys) + tid;
      atomic_key->store(empty_key, cuda::std::memory_order_relaxed);
    }
  }
}

template <typename K, typename V, typename S, typename Tidx, int TILE_SIZE = 8>
void gpu_boolean_mask(size_t grid_size, size_t block_size, const bool* masks,
                      size_t n, size_t* n_evicted, Tidx* offsets,
                      K* __restrict keys, V* __restrict values,
                      S* __restrict scores, size_t dim, cudaStream_t stream) {
  size_t n_offsets = (n + TILE_SIZE - 1) / TILE_SIZE;
  gpu_cell_count<Tidx, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, offsets, n, n_evicted);
#if THRUST_VERSION >= 101600
  auto policy = thrust::cuda::par_nosync.on(stream);
#else
  auto policy = thrust::cuda::par.on(stream);
#endif
  thrust::device_ptr<Tidx> d_src(offsets);
  thrust::device_ptr<Tidx> d_dest(offsets);
  thrust::exclusive_scan(policy, d_src, d_src + n_offsets, d_dest);
  gpu_select_kvm_kernel<K, V, S, Tidx, TILE_SIZE>
      <<<grid_size, block_size, 0, stream>>>(masks, n, offsets, keys, values,
                                             scores, dim);
}

template <typename V>
__global__ void gather_values_kernel(const size_t n, const size_t dim,
                                     const size_t* __restrict order1,
                                     const size_t* __restrict order2,
                                     const V* __restrict values,
                                     V* __restrict unique_values) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > n) {
    return;
  }

  size_t idx = order1[order2[tid]];
  for (size_t i = 0; i < dim; ++i) {
    unique_values[tid * dim + i] = values[idx * dim + i];
  }
}

template <typename V, typename S>
__global__ void gather_scores_values_kernel(const size_t n, const size_t dim,
                                            const size_t* __restrict order1,
                                            const size_t* __restrict order2,
                                            const S* __restrict scores,
                                            S* __restrict unique_scores,
                                            const V* __restrict values,
                                            V* __restrict unique_values) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > n) {
    return;
  }

  size_t idx = order1[order2[tid]];
  for (size_t i = 0; i < dim; ++i) {
    unique_values[tid * dim + i] = values[idx * dim + i];
  }
  unique_scores[tid + i] = scores[idx + i];
}

template <typename K, typename V>
void unique_kv(
    const thrust::device_execution_policy& policy,
    size_t n, size_t dim,
    const K* keys,     // (n)
    const V* values,   // (n * dim)
    K* unique_keys,    // (n)
    V* unique_values,  //(n * dim),
    size_t* unique_size, cudaStream_t stream) {
  cudaMemcpyAsync(unique_keys, keys, sizeof(K) * n, cudaMemcpyDeviceToDevice,
                  stream);

  size_t* order1;
  size_t* order2;
  cudaMallocAsync(&order1, sizeof(size_t) * n, stream);
  cudaMallocAsync(&order2, sizeof(size_t) * n, stream);

  thrust::sequence(policy, order1, order1 + n);
  thrust::sequence(policy, order2, order2 + n);

  thrust::sort_by_key(policy, unique_keys, unique_keys + n, order1);

  thrust::pair<K*, size_t*> end =
      thrust::unique_by_key(policy, unique_keys, unique_keys + n, order2);
  *unique_size = end.first - unique_keys;

  // Mapping from `order2` to `order1`
  static constexpr size_t BLOCK_SIZE = 1024;
  size_t grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  MappingKernel<V><<<grid_size, BLOCK_SIZE, 0, stream>>>(
      *unique_size, dim, order1, order2, values, unique_values);

  cudaFreeAsync(order1, stream);
  cudaFreeAsync(order2, stream);
}

template <typename K, typename V, typename S>
void unique_kvs(
    const thrust::device_execution_policy& policy,
    size_t n, size_t dim,
    const K* keys,     // (n)
    const V* values,   // (n * dim)
    const S* scores,   // (n)
    K* unique_keys,    // (n)
    V* unique_values,  // (n * dim)
    S* unique_scores,  // (n)
    size_t* unique_size, cudaStream_t stream) {
  cudaMemcpyAsync(unique_keys, keys, sizeof(K) * n, cudaMemcpyDeviceToDevice,
                  stream);

  size_t* order1;
  size_t* order2;
  cudaMallocAsync(&order1, sizeof(size_t) * n, stream);
  cudaMallocAsync(&order2, sizeof(size_t) * n, stream);

  thrust::sequence(policy, order1, order1 + n);
  thrust::sequence(policy, order2, order2 + n);

  thrust::sort_by_key(policy, unique_keys, unique_keys + n, order1);

  thrust::pair<K*, size_t*> end =
      thrust::unique_by_key(policy, unique_keys, unique_keys + n, order2);
  *unique_size = end.first - unique_keys;

  // Mapping from `order2` to `order1`
  static constexpr size_t BLOCK_SIZE = 1024;
  size_t grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  gather_values_kernel<V><<<grid_size, BLOCK_SIZE, 0, stream>>>(
      *unique_size, dim, order1, order2, values, scores, unique_values,
      unique_scores);

  cudaFreeAsync(order1, stream);
  cudaFreeAsync(order2, stream);
}

}  // namespace merlin
}  // namespace nv
