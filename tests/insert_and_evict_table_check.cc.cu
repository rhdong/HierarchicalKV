/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <stdio.h>
#include <array>
#include <map>
#include <chrono>
#include <cmath>
#include <cstdint>
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

template <typename K, typename V, typename M>
bool CheckInsertAndEvict(Table* table, K* keys, V* values, M* metas,
                         K* evicted_keys, V* evicted_values, M* evicted_metas,
                         size_t len, cudaStream_t stream) {
  std::map<i64, test_util::ValueArray<f32, dim>> map_before_insert;
  std::map<i64, test_util::ValueArray<f32, dim>> map_after_insert;
  K* h_tmp_keys = nullptr;
  V* h_tmp_values = nullptr;
  M* h_tmp_metas = nullptr;

  K* d_tmp_keys = nullptr;
  V* d_tmp_values = nullptr;
  M* d_tmp_metas = nullptr;

  size_t table_size_before = table->size(stream);
  size_t cap = table_size_before + len;

  CUDA_CHECK(cudaMallocAsync(&d_tmp_keys, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_keys, 0, cap * sizeof(K), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_values, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_values, 0, cap * dim * sizeof(V), stream));
  CUDA_CHECK(cudaMallocAsync(&d_tmp_metas, cap * sizeof(M), stream));
  CUDA_CHECK(cudaMemsetAsync(d_tmp_metas, 0, cap * sizeof(M), stream));
  h_tmp_keys = (K*) malloc(cap * sizeof(K));
  h_tmp_values = (V*) malloc(cap * dim * sizeof(V));
  h_tmp_metas = (M*) malloc(cap * sizeof(M));

  size_t table_size_verify0 = table->export_batch(table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_metas, stream);
  if (table_size_before != table_size_verify0) {
    fprintf(stderr, "[FATAL] dump error.\n");
    exit(1);
  }
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys, table_size_before * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values, table_size_before * dim * sizeof(V), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas, d_tmp_metas, table_size_before * sizeof(M), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_before, keys, len * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_before * dim, values, len * dim * sizeof(V), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas + table_size_before, metas, len * sizeof(M), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < cap; i++) {
    test_util::ValueArray<V, dim>* vec = reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values + i * dim);
    map_before_insert[h_tmp_keys[i]] = *vec;
  }


  auto start = std::chrono::steady_clock::now();
  size_t filtered_len = table->insert_and_evict(len, keys, values, nullptr, evicted_keys, evicted_values, evicted_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start);

  float dur = diff.count();

  std::cout << "dur=" << dur << std::endl;

  size_t table_size_after = table->size(stream);
  size_t table_size_verify1 = table->export_batch(table->capacity(), 0, d_tmp_keys, d_tmp_values, d_tmp_metas, stream);
  if (table_size_verify1 != table_size_after) {
    fprintf(stderr, "");
    exit(1);
  }

  size_t new_cap = table_size_after + filtered_len;
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys, d_tmp_keys, table_size_after * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values, d_tmp_values, table_size_after * dim * sizeof(V), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas, d_tmp_metas, table_size_after * sizeof(M), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_keys + table_size_after, evicted_keys, filtered_len * sizeof(K), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_values + table_size_after * dim, evicted_values, filtered_len * dim * sizeof(V), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_tmp_metas + table_size_after, evicted_metas, filtered_len * sizeof(M), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int64_t new_cap_i64 = (int64_t)new_cap;
  for (int64_t i = new_cap_i64 - 1; i >= 0; i--) {
    test_util::ValueArray<V, dim>* vec = reinterpret_cast<test_util::ValueArray<V, dim>*>(h_tmp_values + i * dim);
    map_after_insert[h_tmp_keys[i]] = *vec;
  }

  size_t key_miss_cnt = 0;
  size_t value_diff_cnt = 0;
  for (auto& it : map_before_insert) {
    if (map_after_insert.find(it.first) == map_after_insert.end()) {
      ++key_miss_cnt;
      continue;
    }
    test_util::ValueArray<V, dim>& vec0 = it.second;
    test_util::ValueArray<V, dim>& vec1 = map_after_insert.at(it.first);
    for (size_t j = 0; j < dim; j++) {
      if (vec0[j] != vec1[j]) {
        ++value_diff_cnt;
        break;
      }
    }
  }
  std::cout << "Check evict behavior got key_miss_cnt: " << key_miss_cnt
            << ", and value_diff_cnt: " << value_diff_cnt
            << ", while table_size_before: " << table_size_before
            << ", while table_size_after: " << table_size_after
            << ", while len: " << len << std::endl;

  CUDA_CHECK(cudaFreeAsync(d_tmp_keys, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_values, stream));
  CUDA_CHECK(cudaFreeAsync(d_tmp_metas, stream));
  free(h_tmp_keys);
  free(h_tmp_values);
  free(h_tmp_metas);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return (value_diff_cnt == 0);
}

void test_insert_and_evict_table_check() {
  TableOptions opt;

  // table setting

  // numeric setting
  const size_t U = 524288;
  const size_t init_capacity = 1024;
  const size_t B = 524288 + 13;

  opt.max_capacity = U;
  opt.init_capacity = init_capacity;
  opt.max_hbm_for_vectors = U * dim * sizeof(f32);
  opt.evict_strategy = nv::merlin::EvictStrategy::kLru;
  opt.dim = dim;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(opt);

  test_util::KVMSBuffer<i64, f32, u64> evict_buffer;
  evict_buffer.Reserve(B, dim, stream);
  evict_buffer.ToZeros(stream);

  test_util::KVMSBuffer<i64, f32, u64> data_buffer;
  data_buffer.Reserve(B, dim, stream);

  size_t len = B;
  size_t offset = 0;
  u64 meta = 0;
  while (true) {
    test_util::create_random_keys<i64, u64, f32, dim>(data_buffer.keys_ptr(false), data_buffer.metas_ptr(false), data_buffer.values_ptr(false), (int)B);
    data_buffer.SyncData(true, stream);

    printf("----> check p0\n");
    bool if_ok = CheckInsertAndEvict<i64, f32, u64>(table.get(),
                        data_buffer.keys_ptr(), data_buffer.values_ptr(), data_buffer.metas_ptr(),
                        evict_buffer.keys_ptr(), evict_buffer.values_ptr(), evict_buffer.metas_ptr(),
                        B, stream);
    printf("----> check p1\n");

    offset += B;
    meta += 1;
    if(!if_ok) break;
  }
}

int main() {
  test_insert_and_evict_table_check();
  return 0;
}
