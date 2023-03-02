/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

uint64_t getTimestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

constexpr size_t DIM = 16;

template <class K, class M, class V>
void create_random_keys(K* h_keys, M* h_metas, V* h_vectors, int KEY_NUM) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    numbers.insert(distr(eng));
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_metas != nullptr) {
      h_metas[i] = num;
    }
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < DIM; j++) {
        h_vectors[i * DIM + j] = static_cast<V>(num * 0.00001);
      }
    }
    i++;
  }
}

template <class K, class M, class V>
void create_continuous_keys(K* h_keys, M* h_metas, V* h_vectors, int KEY_NUM,
                            K start = 1) {
  for (K i = 0; i < KEY_NUM; i++) {
    h_keys[i] = start + static_cast<K>(i);
    h_metas[i] = h_keys[i];
    if (h_vectors != nullptr) {
      for (size_t j = 0; j < DIM; j++) {
        h_vectors[i * DIM + j] = static_cast<V>(h_keys[i] * 0.00001);
      }
    }
  }
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

template <class K, class M, class V>
void create_keys_in_one_buckets(K* h_keys, M* h_metas, V* h_vectors,
                                int KEY_NUM, int capacity,
                                int bucket_max_size = 128, int bucket_idx = 0,
                                K min = 0,
                                K max = static_cast<K>(0xFFFFFFFFFFFFFFFD)) {
  std::unordered_set<K> numbers;
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<K> distr;
  K candidate;
  K hashed_key;
  size_t global_idx;
  size_t bkt_idx;
  int i = 0;

  while (numbers.size() < KEY_NUM) {
    candidate = (distr(eng) % (max - min)) + min;
    hashed_key = Murmur3HashHost(candidate);
    global_idx = hashed_key & (capacity - 1);
    bkt_idx = global_idx / bucket_max_size;
    if (bkt_idx == bucket_idx) {
      numbers.insert(candidate);
    }
  }
  for (const K num : numbers) {
    h_keys[i] = num;
    if (h_metas != nullptr) {
      h_metas[i] = num;
    }
    for (size_t j = 0; j < DIM; j++) {
      *(h_vectors + i * DIM + j) = static_cast<V>(num * 0.00001);
    }
    i++;
  }
}

using K = uint64_t;
using V = float;
using M = uint64_t;
using Table = nv::merlin::HashTable<K, V, M>;
using TableOptions = nv::merlin::HashTableOptions;

template <class K, class M>
__forceinline__ __device__ bool erase_if_pred(const K& key, M& meta,
                                              const K& pattern,
                                              const M& threshold) {
  return ((key & 0x7f > pattern) && (meta > threshold));
}

template <class K, class M>
__device__ Table::Pred EraseIfPred = erase_if_pred<K, M>;

template <class K, class M>
__forceinline__ __device__ bool export_if_pred(const K& key, M& meta,
                                               const K& pattern,
                                               const M& threshold) {
  return meta > threshold;
}

template <class K, class M>
__device__ Table::Pred ExportIfPred = export_if_pred<K, M>;

void test_basic(size_t max_hbm_for_vectors, bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

  create_random_keys<K, M, V>(h_keys, h_metas, h_vectors, KEY_NUM);

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_new_vectors;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_new_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
      ASSERT_EQ(h_metas[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<V>(h_keys[i] * 0.00001));
      }
    }
    CUDA_CHECK(cudaMemset(d_new_vectors, 2, KEY_NUM * sizeof(V) * options.dim));
    table->insert_or_assign(KEY_NUM, d_keys,
                            reinterpret_cast<float*>(d_new_vectors), d_metas,
                            stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    CUDA_CHECK(cudaMemset(d_new_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, reinterpret_cast<float*>(d_new_vectors),
                d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_new_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    found_num = 0;
    uint32_t i_value = 0x2020202;
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  *(reinterpret_cast<float*>(&i_value)));
      }
    }
    ASSERT_EQ(found_num, KEY_NUM);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_found, d_metas,
                           stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    table->erase(KEY_NUM >> 1, d_keys, stream);
    size_t total_size_after_erase = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_erase, total_size >> 1);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemset(d_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

    table->find(KEY_NUM, d_keys, d_vectors, d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    found_num = 0;
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
      ASSERT_EQ(h_metas[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<V>(h_keys[i] * 0.00001));
      }
    }
    ASSERT_EQ(found_num, KEY_NUM);

    CUDA_CHECK(cudaMemset(d_keys, 0, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMemset(d_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                       d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ASSERT_EQ(dump_counter, KEY_NUM);
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      ASSERT_EQ(h_metas[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<V>(h_keys[i] * 0.00001));
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_new_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_basic_when_full(size_t max_hbm_for_vectors,
                          bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 1 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

  create_random_keys<K, M, V>(h_keys, h_metas, nullptr, KEY_NUM);

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_vectors, 1, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t total_size_after_insert = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    table->erase(KEY_NUM, d_keys, stream);
    size_t total_size_after_erase = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_erase, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint64_t total_size_after_reinsert = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_insert, total_size_after_reinsert);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_erase_if_pred(size_t max_hbm_for_vectors, bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr uint64_t BUCKET_MAX_SIZE = 128;

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  bool* d_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    create_keys_in_one_buckets<K, M, float>(h_keys, h_metas, h_vectors, KEY_NUM,
                                            INIT_CAPACITY);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

    K pattern = 100;
    M threshold = 0;
    size_t erase_num =
        table->erase_if(EraseIfPred<K, M>, pattern, threshold, stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ((erase_num + total_size), BUCKET_MAX_SIZE);

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_EQ(h_metas[i], h_keys[i]);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<V>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, (BUCKET_MAX_SIZE - erase_num));

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_rehash(size_t max_hbm_for_vectors, bool use_constant_memory) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = BUCKET_MAX_SIZE;
  constexpr uint64_t MAX_CAPACITY = 4 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = BUCKET_MAX_SIZE * 2;
  constexpr uint64_t TEST_TIMES = 100;
  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    create_keys_in_one_buckets<K, M, float>(h_keys, h_metas, h_vectors, KEY_NUM,
                                            INIT_CAPACITY, BUCKET_MAX_SIZE);
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    ASSERT_EQ(total_size, KEY_NUM);

    dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                       d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(dump_counter, KEY_NUM);

    table->reserve(MAX_CAPACITY, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(table->capacity(), MAX_CAPACITY);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(BUCKET_MAX_SIZE, d_keys, d_vectors, d_found, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < BUCKET_MAX_SIZE; i++) {
      if (h_found[i]) {
        found_num++;
        ASSERT_EQ(h_metas[i], h_keys[i]);
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<V>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, BUCKET_MAX_SIZE);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_rehash_on_big_batch(size_t max_hbm_for_vectors,
                              bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 1024;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024;
  constexpr uint64_t INIT_KEY_NUM = 1024;
  constexpr uint64_t KEY_NUM = 2048;
  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = 128;
  options.max_load_factor = 0.6;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  uint64_t expected_size = 0;
  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  create_random_keys<K, M, V>(h_keys, h_metas, h_vectors, KEY_NUM);

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  total_size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(total_size, 0);

  table->insert_or_assign(INIT_KEY_NUM, d_keys, d_vectors, d_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  expected_size = INIT_KEY_NUM;

  total_size = table->size(stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  ASSERT_EQ(total_size, expected_size);
  ASSERT_EQ(table->capacity(), (INIT_CAPACITY * 2));

  table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  expected_size = KEY_NUM;

  total_size = table->size(stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  ASSERT_EQ(total_size, expected_size);
  ASSERT_EQ(table->capacity(), KEY_NUM * 4);

  dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                     d_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(dump_counter, expected_size);

  CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
  table->find(KEY_NUM, d_keys, d_vectors, d_found, d_metas, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int found_num = 0;

  CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(
      cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < KEY_NUM; i++) {
    if (h_found[i]) {
      found_num++;
      ASSERT_EQ(h_metas[i], h_keys[i]);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<V>(h_keys[i] * 0.00001));
      }
    }
  }
  ASSERT_EQ(found_num, KEY_NUM);

  table->clear(stream);
  total_size = table->size(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_EQ(total_size, 0);
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_dynamic_rehash_on_multi_threads(size_t max_hbm_for_vectors,
                                          bool use_constant_memory) {
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;
  constexpr uint64_t INIT_CAPACITY = 4 * 1024;
  constexpr uint64_t MAX_CAPACITY = 16 * 1024 * INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 256;
  constexpr uint64_t THREAD_N = 8;

  std::vector<std::thread> threads;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_load_factor = 0.50f;
  options.max_bucket_size = BUCKET_MAX_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kLru;
  options.use_constant_memory = use_constant_memory;

  std::shared_ptr<Table> table = std::make_shared<Table>();
  table->init(options);

  auto worker_function = [&table, KEY_NUM, options](int task_n) {
    K* h_keys;
    V* h_vectors;
    bool* h_found;

    size_t current_capacity = table->capacity();

    CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

    K* d_keys;
    V* d_vectors;
    bool* d_found;

    CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    while (table->capacity() < MAX_CAPACITY) {
      create_random_keys<K, M, V>(h_keys, nullptr, h_vectors, KEY_NUM);
      CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

      table->insert_or_assign(KEY_NUM, d_keys, d_vectors, nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
      table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      int found_num = 0;

      CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
      CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
      CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                            cudaMemcpyDeviceToHost));

      CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                            KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < KEY_NUM; i++) {
        if (h_found[i]) {
          found_num++;
          for (int j = 0; j < options.dim; j++) {
            ASSERT_EQ(h_vectors[i * options.dim + j],
                      static_cast<V>(h_keys[i] * 0.00001));
          }
        }
      }
      ASSERT_EQ(found_num, KEY_NUM);
      if (task_n == 0 && current_capacity != table->capacity()) {
        std::cout << "[test_dynamic_rehash_on_multi_threads] The capacity "
                     "changed from "
                  << current_capacity << " to " << table->capacity()
                  << std::endl;
        current_capacity = table->capacity();
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeHost(h_keys));
    CUDA_CHECK(cudaFreeHost(h_found));
    CUDA_CHECK(cudaFreeHost(h_vectors));

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_vectors));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaCheckError();
  };

  for (int i = 0; i < THREAD_N; ++i)
    threads.emplace_back(std::thread(worker_function, i));

  for (auto& th : threads) {
    th.join();
  }
  ASSERT_EQ(table->capacity(), MAX_CAPACITY);
}

void test_export_batch_if(size_t max_hbm_for_vectors,
                          bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 256UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 128UL;
  constexpr uint64_t TEST_TIMES = 1;
  constexpr uint64_t BUCKET_MAX_SIZE = 128ul;

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  size_t h_dump_counter = 0;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  std::unique_ptr<Table> table = std::make_unique<Table>();
  table->init(options);

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  bool* d_found;
  size_t* d_dump_counter;
  int found_num = 0;
  bool* h_found;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_dump_counter, sizeof(size_t)));

  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    create_random_keys<K, M, V>(h_keys, h_metas, h_vectors, KEY_NUM);

    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyHostToDevice));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    CUDA_CHECK(cudaMemset(h_found, 0, KEY_NUM * sizeof(bool)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_vectors, 0, KEY_NUM * sizeof(V) * options.dim));
    table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));
    found_num = 0;
    for (int i = 0; i < BUCKET_MAX_SIZE; i++) {
      if (h_found[i]) {
        found_num++;
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors[i * options.dim + j],
                    static_cast<V>(h_keys[i] * 0.00001));
        }
      }
    }
    ASSERT_EQ(found_num, KEY_NUM);

    K pattern = 100;
    M threshold = h_metas[size_t(KEY_NUM / 2)];

    table->export_batch_if(ExportIfPred<K, M>, pattern, threshold,
                           table->capacity(), 0, d_dump_counter, d_keys,
                           d_vectors, d_metas, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&h_dump_counter, d_dump_counter, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    size_t expected_export_count = 0;
    for (int i = 0; i < KEY_NUM; i++) {
      if (h_metas[i] > threshold) expected_export_count++;
    }
    ASSERT_EQ(expected_export_count, h_dump_counter);

    CUDA_CHECK(cudaMemset(h_keys, 0, KEY_NUM * sizeof(K)));
    CUDA_CHECK(cudaMemset(h_metas, 0, KEY_NUM * sizeof(M)));
    CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, KEY_NUM * sizeof(K),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_metas, d_metas, KEY_NUM * sizeof(M),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors,
                          KEY_NUM * sizeof(V) * options.dim,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_dump_counter; i++) {
      ASSERT_GT(h_metas[i], threshold);
      for (int j = 0; j < options.dim; j++) {
        ASSERT_EQ(h_vectors[i * options.dim + j],
                  static_cast<V>(h_keys[i] * 0.00001));
      }
    }

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaFree(d_dump_counter));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_basic_for_cpu_io(bool use_constant_memory) {
  constexpr uint64_t INIT_CAPACITY = 64 * 1024 * 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t KEY_NUM = 1 * 1024 * 1024UL;
  constexpr uint64_t TEST_TIMES = 1;

  K* h_keys;
  M* h_metas;
  V* h_vectors;
  bool* h_found;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(0);
  options.io_by_cpu = true;
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  CUDA_CHECK(cudaMallocHost(&h_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMallocHost(&h_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMallocHost(&h_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(cudaMemset(h_vectors, 0, KEY_NUM * sizeof(V) * options.dim));

  create_random_keys<K, M, V>(h_keys, h_metas, nullptr, KEY_NUM);

  K* d_keys;
  M* d_metas = nullptr;
  V* d_vectors;
  V* d_def_val;
  V** d_vectors_ptr;
  bool* d_found;
  size_t dump_counter = 0;

  CUDA_CHECK(cudaMalloc(&d_keys, KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas, KEY_NUM * sizeof(M)));
  CUDA_CHECK(cudaMalloc(&d_vectors, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_def_val, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMalloc(&d_vectors_ptr, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMalloc(&d_found, KEY_NUM * sizeof(bool)));

  CUDA_CHECK(
      cudaMemcpy(d_keys, h_keys, KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_metas, h_metas, KEY_NUM * sizeof(M),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemset(d_vectors, 1, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_def_val, 2, KEY_NUM * sizeof(V) * options.dim));
  CUDA_CHECK(cudaMemset(d_vectors_ptr, 0, KEY_NUM * sizeof(V*)));
  CUDA_CHECK(cudaMemset(d_found, 0, KEY_NUM * sizeof(bool)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint64_t total_size = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    CUDA_CHECK(cudaMemset(d_vectors, 2, KEY_NUM * sizeof(V) * options.dim));
    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    table->find(KEY_NUM, d_keys, d_vectors, d_found, nullptr, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int found_num = 0;
    CUDA_CHECK(cudaMemcpy(h_found, d_found, KEY_NUM * sizeof(bool),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < KEY_NUM; i++) {
      if (h_found[i]) found_num++;
    }
    ASSERT_EQ(found_num, KEY_NUM);

    table->accum_or_assign(KEY_NUM, d_keys, d_vectors, d_found, d_metas,
                           stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, KEY_NUM);

    table->erase(KEY_NUM >> 1, d_keys, stream);
    size_t total_size_after_erase = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size_after_erase, total_size >> 1);

    table->clear(stream);
    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    table->insert_or_assign(KEY_NUM, d_keys, d_vectors, d_metas, stream);

    dump_counter = table->export_batch(table->capacity(), 0, d_keys, d_vectors,
                                       d_metas, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(dump_counter, KEY_NUM);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, KEY_NUM * sizeof(V) * options.dim,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFreeHost(h_keys));
  CUDA_CHECK(cudaFreeHost(h_metas));
  CUDA_CHECK(cudaFreeHost(h_found));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_metas));
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaFree(d_def_val));
  CUDA_CHECK(cudaFree(d_vectors_ptr));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_lru_basic(size_t max_hbm_for_vectors,
                                   bool use_constant_memory) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 4;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kLru;
  options.use_constant_memory = use_constant_memory;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<M, BASE_KEY_NUM> h_metas_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<M, TEST_KEY_NUM> h_metas_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<M, TEMP_KEY_NUM> h_metas_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  M* d_metas_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas_temp, TEMP_KEY_NUM * sizeof(M)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  create_keys_in_one_buckets<K, M, V>(
      h_keys_base.data(), h_metas_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  create_keys_in_one_buckets<K, M, V>(h_keys_test.data(), h_metas_test.data(),
                                      h_vectors_test.data(), TEST_KEY_NUM,
                                      INIT_CAPACITY, BUCKET_MAX_SIZE, 1,
                                      0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFD);

  h_keys_test[2] = h_keys_base[72];
  h_keys_test[3] = h_keys_base[73];

  for (int i = 0; i < options.dim; i++) {
    h_vectors_test[2 * options.dim + i] = h_vectors_base[72 * options.dim + i];
    h_vectors_test[3 * options.dim + i] = h_vectors_base[73 * options.dim + i];
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_base.data(),
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                              nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<M, BASE_KEY_NUM> h_metas_temp_sorted(h_metas_temp);
      std::sort(h_metas_temp_sorted.begin(), h_metas_temp_sorted.end());

      ASSERT_TRUE(
          (h_metas_temp_sorted == test_util::range<M, TEMP_KEY_NUM>(1)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_test.data(),
                            TEST_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                              nullptr, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            TEMP_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<M, TEST_KEY_NUM> h_metas_temp_sorted;
      int ctr = 0;
      for (int i = 0; i < TEMP_KEY_NUM; i++) {
        if (h_keys_test.end() !=
            std::find(h_keys_test.begin(), h_keys_test.end(), h_keys_temp[i])) {
          ASSERT_GT(h_metas_temp[i], BUCKET_MAX_SIZE);
          h_metas_temp_sorted[ctr++] = h_metas_temp[i];
        } else {
          ASSERT_LE(h_metas_temp[i], BUCKET_MAX_SIZE);
        }
      }
      std::sort(h_metas_temp_sorted.begin(), h_metas_temp_sorted.begin() + ctr);

      ASSERT_TRUE((h_metas_temp_sorted ==
                   test_util::range<M, TEST_KEY_NUM>(BUCKET_MAX_SIZE + 1)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_metas_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_basic(size_t max_hbm_for_vectors,
                                          bool use_constant_memory) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 128;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 128;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<M, BASE_KEY_NUM> h_metas_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<M, TEST_KEY_NUM> h_metas_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<M, TEMP_KEY_NUM> h_metas_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  M* d_metas_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas_temp, TEMP_KEY_NUM * sizeof(M)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  create_keys_in_one_buckets<K, M, V>(
      h_keys_base.data(), h_metas_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  const M base_meta_start = 1000;
  for (int i = 0; i < BASE_KEY_NUM; i++) {
    h_metas_base[i] = base_meta_start + i;
  }

  create_keys_in_one_buckets<K, M, V>(h_keys_test.data(), h_metas_test.data(),
                                      h_vectors_test.data(), TEST_KEY_NUM,
                                      INIT_CAPACITY, BUCKET_MAX_SIZE, 1,
                                      0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFD);
  const M test_meta_start = base_meta_start + BASE_KEY_NUM;
  for (int i = 0; i < TEST_KEY_NUM; i++) {
    h_metas_test[i] = test_meta_start + i;
  }
  for (int i = 64; i < TEST_KEY_NUM; i++) {
    h_keys_test[i] = h_keys_base[i];
    for (int j = 0; j < options.dim; j++) {
      h_vectors_test[i * options.dim + j] = h_vectors_base[i * options.dim + j];
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_base.data(),
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                              d_metas_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<M, BASE_KEY_NUM> h_metas_temp_sorted(h_metas_temp);
      std::sort(h_metas_temp_sorted.begin(), h_metas_temp_sorted.end());

      ASSERT_TRUE((h_metas_temp_sorted ==
                   test_util::range<M, TEMP_KEY_NUM>(base_meta_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_test.data(),
                            TEST_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                              d_metas_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            TEMP_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<M, TEST_KEY_NUM> h_metas_temp_sorted(h_metas_temp);
      std::sort(h_metas_temp_sorted.begin(), h_metas_temp_sorted.end());

      ASSERT_TRUE((h_metas_temp_sorted ==
                   test_util::range<M, TEST_KEY_NUM>(test_meta_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_metas_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_advanced(size_t max_hbm_for_vectors,
                                             bool use_constant_memory) {
  constexpr uint64_t BUCKET_NUM = 8UL;
  constexpr uint64_t BUCKET_MAX_SIZE = 128UL;
  constexpr uint64_t INIT_CAPACITY = BUCKET_NUM * BUCKET_MAX_SIZE;  // 1024UL;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t BASE_KEY_NUM = BUCKET_MAX_SIZE;
  constexpr uint64_t TEST_KEY_NUM = 8;
  constexpr uint64_t TEMP_KEY_NUM = std::max(BASE_KEY_NUM, TEST_KEY_NUM);
  constexpr uint64_t TEST_TIMES = 256;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  std::array<K, BASE_KEY_NUM> h_keys_base;
  std::array<M, BASE_KEY_NUM> h_metas_base;
  std::array<V, BASE_KEY_NUM * DIM> h_vectors_base;

  std::array<K, TEST_KEY_NUM> h_keys_test;
  std::array<M, TEST_KEY_NUM> h_metas_test;
  std::array<V, TEST_KEY_NUM * DIM> h_vectors_test;

  std::array<K, TEMP_KEY_NUM> h_keys_temp;
  std::array<M, TEMP_KEY_NUM> h_metas_temp;
  std::array<V, TEMP_KEY_NUM * DIM> h_vectors_temp;

  K* d_keys_temp;
  M* d_metas_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMalloc(&d_keys_temp, TEMP_KEY_NUM * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas_temp, TEMP_KEY_NUM * sizeof(M)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, TEMP_KEY_NUM * sizeof(V) * options.dim));

  create_keys_in_one_buckets<K, M, V>(
      h_keys_base.data(), h_metas_base.data(), h_vectors_base.data(),
      BASE_KEY_NUM, INIT_CAPACITY, BUCKET_MAX_SIZE, 1, 0, 0x3FFFFFFFFFFFFFFF);

  const M base_meta_start = 1000;
  for (int i = 0; i < BASE_KEY_NUM; i++) {
    h_metas_base[i] = base_meta_start + i;
  }

  create_keys_in_one_buckets<K, M, V>(h_keys_test.data(), h_metas_test.data(),
                                      h_vectors_test.data(), TEST_KEY_NUM,
                                      INIT_CAPACITY, BUCKET_MAX_SIZE, 1,
                                      0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFD);

  h_keys_test[4] = h_keys_base[72];
  h_keys_test[5] = h_keys_base[73];
  h_keys_test[6] = h_keys_base[74];
  h_keys_test[7] = h_keys_base[75];

  // replace four new keys to lower metas, would not be inserted.
  h_metas_test[0] = 20;
  h_metas_test[1] = 78;
  h_metas_test[2] = 97;
  h_metas_test[3] = 98;

  // replace three exist keys to new metas, just refresh the meta for them.
  h_metas_test[4] = 99;
  h_metas_test[5] = 1010;
  h_metas_test[6] = 1020;
  h_metas_test[7] = 1035;

  for (int i = 4; i < TEST_KEY_NUM; i++) {
    for (int j = 0; j < options.dim; j++) {
      h_vectors_test[i * options.dim + j] =
          static_cast<V>(h_keys_test[i] * 0.00001);
    }
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t dump_counter = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base.data(),
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_base.data(),
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base.data(),
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(BASE_KEY_NUM, d_keys_temp, d_vectors_temp,
                              d_metas_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            BASE_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            BASE_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            BASE_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      std::array<M, BASE_KEY_NUM> h_metas_temp_sorted(h_metas_temp);
      std::sort(h_metas_temp_sorted.begin(), h_metas_temp_sorted.end());

      ASSERT_TRUE((h_metas_temp_sorted ==
                   test_util::range<M, TEMP_KEY_NUM>(base_meta_start)));
      for (int i = 0; i < dump_counter; i++) {
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }

    {
      CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_test.data(),
                            TEST_KEY_NUM * sizeof(K), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_test.data(),
                            TEST_KEY_NUM * sizeof(M), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_test.data(),
                            TEST_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyHostToDevice));
      table->insert_or_assign(TEST_KEY_NUM, d_keys_temp, d_vectors_temp,
                              d_metas_temp, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_EQ(total_size, BUCKET_MAX_SIZE);

      dump_counter = table->export_batch(table->capacity(), 0, d_keys_temp,
                                         d_vectors_temp, d_metas_temp, stream);
      ASSERT_EQ(dump_counter, BUCKET_MAX_SIZE);

      CUDA_CHECK(cudaMemcpy(h_keys_temp.data(), d_keys_temp,
                            TEMP_KEY_NUM * sizeof(K), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp.data(), d_metas_temp,
                            TEMP_KEY_NUM * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp.data(), d_vectors_temp,
                            TEMP_KEY_NUM * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      for (int i = 0; i < TEST_KEY_NUM; i++) {
        if (i < 4) {
          ASSERT_EQ(h_keys_temp.end(),
                    std::find(h_keys_temp.begin(), h_keys_temp.end(),
                              h_keys_test[i]));
        } else {
          ASSERT_NE(h_keys_temp.end(),
                    std::find(h_keys_temp.begin(), h_keys_temp.end(),
                              h_keys_test[i]));
        }
      }
      for (int i = 0; i < TEMP_KEY_NUM; i++) {
        if (h_keys_temp[i] == h_keys_test[4])
          ASSERT_EQ(h_metas_temp[i], h_metas_test[4]);
        if (h_keys_temp[i] == h_keys_test[5])
          ASSERT_EQ(h_metas_temp[i], h_metas_test[5]);
        if (h_keys_temp[i] == h_keys_test[6])
          ASSERT_EQ(h_metas_temp[i], h_metas_test[6]);
        if (h_keys_temp[i] == h_keys_test[7])
          ASSERT_EQ(h_metas_temp[i], h_metas_test[7]);

        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<V>(h_keys_temp[i] * 0.00001));
        }
      }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_metas_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

void test_evict_strategy_customized_correct_rate(size_t max_hbm_for_vectors,
                                                 bool use_constant_memory) {
  constexpr uint64_t BATCH_SIZE = 1024 * 1024ul;
  constexpr uint64_t STEPS = 128;
  constexpr uint64_t MAX_BUCKET_SIZE = 128;
  constexpr uint64_t INIT_CAPACITY = BATCH_SIZE * STEPS;
  constexpr uint64_t MAX_CAPACITY = INIT_CAPACITY;
  constexpr uint64_t TEST_TIMES = 1;
  float expected_correct_rate = 0.964;
  const int rounds = 3;

  TableOptions options;

  options.init_capacity = INIT_CAPACITY;
  options.max_capacity = MAX_CAPACITY;
  options.dim = DIM;
  options.max_bucket_size = MAX_BUCKET_SIZE;
  options.max_hbm_for_vectors = nv::merlin::GB(max_hbm_for_vectors);
  options.evict_strategy = nv::merlin::EvictStrategy::kCustomized;
  options.use_constant_memory = use_constant_memory;

  K* h_keys_base;
  M* h_metas_base;
  V* h_vectors_base;

  K* h_keys_temp;
  M* h_metas_temp;
  V* h_vectors_temp;

  K* d_keys_temp;
  M* d_metas_temp = nullptr;
  V* d_vectors_temp;

  CUDA_CHECK(cudaMallocHost(&h_keys_base, BATCH_SIZE * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas_base, BATCH_SIZE * sizeof(M)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors_base, BATCH_SIZE * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMallocHost(&h_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMallocHost(&h_metas_temp, MAX_CAPACITY * sizeof(M)));
  CUDA_CHECK(
      cudaMallocHost(&h_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));

  CUDA_CHECK(cudaMalloc(&d_keys_temp, MAX_CAPACITY * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_metas_temp, MAX_CAPACITY * sizeof(M)));
  CUDA_CHECK(
      cudaMalloc(&d_vectors_temp, MAX_CAPACITY * sizeof(V) * options.dim));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  size_t total_size = 0;
  size_t global_start_key = 100000;
  for (int i = 0; i < TEST_TIMES; i++) {
    std::unique_ptr<Table> table = std::make_unique<Table>();
    table->init(options);
    size_t start_key = global_start_key;

    total_size = table->size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ASSERT_EQ(total_size, 0);

    for (int r = 0; r < rounds; r++) {
      size_t expected_min_key = global_start_key + INIT_CAPACITY * r;
      size_t expected_max_key = global_start_key + INIT_CAPACITY * (r + 1) - 1;
      size_t expected_table_size =
          (r == 0) ? size_t(expected_correct_rate * INIT_CAPACITY)
                   : INIT_CAPACITY;

      for (int s = 0; s < STEPS; s++) {
        create_continuous_keys<K, M, V>(h_keys_base, h_metas_base,
                                        h_vectors_base, BATCH_SIZE, start_key);
        start_key += BATCH_SIZE;

        CUDA_CHECK(cudaMemcpy(d_keys_temp, h_keys_base, BATCH_SIZE * sizeof(K),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_metas_temp, h_metas_base,
                              BATCH_SIZE * sizeof(M), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vectors_temp, h_vectors_base,
                              BATCH_SIZE * sizeof(V) * options.dim,
                              cudaMemcpyHostToDevice));
        table->insert_or_assign(BATCH_SIZE, d_keys_temp, d_vectors_temp,
                                d_metas_temp, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }

      size_t total_size = table->size(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT_GE(total_size, expected_table_size);
      ASSERT_EQ(MAX_CAPACITY, table->capacity());

      size_t dump_counter = table->export_batch(
          MAX_CAPACITY, 0, d_keys_temp, d_vectors_temp, d_metas_temp, stream);

      CUDA_CHECK(cudaMemcpy(h_keys_temp, d_keys_temp, MAX_CAPACITY * sizeof(K),
                            cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_metas_temp, d_metas_temp,
                            MAX_CAPACITY * sizeof(M), cudaMemcpyDefault));
      CUDA_CHECK(cudaMemcpy(h_vectors_temp, d_vectors_temp,
                            MAX_CAPACITY * sizeof(V) * options.dim,
                            cudaMemcpyDefault));

      size_t bigger_meta_counter = 0;
      K max_key = 0;

      for (int i = 0; i < dump_counter; i++) {
        ASSERT_EQ(h_keys_temp[i], h_metas_temp[i]);
        max_key = std::max(max_key, h_keys_temp[i]);
        if (h_metas_temp[i] >= expected_min_key) bigger_meta_counter++;
        for (int j = 0; j < options.dim; j++) {
          ASSERT_EQ(h_vectors_temp[i * options.dim + j],
                    static_cast<float>(h_keys_temp[i] * 0.00001));
        }
      }

      float correct_rate = (bigger_meta_counter * 1.0) / MAX_CAPACITY;
      std::cout << std::setprecision(3) << "[Round " << r << "]"
                << "correct_rate=" << correct_rate << std::endl;
      ASSERT_GE(max_key, expected_max_key);
      ASSERT_GE(correct_rate, expected_correct_rate);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFreeHost(h_keys_base));
  CUDA_CHECK(cudaFreeHost(h_metas_base));
  CUDA_CHECK(cudaFreeHost(h_vectors_base));

  CUDA_CHECK(cudaFreeHost(h_keys_temp));
  CUDA_CHECK(cudaFreeHost(h_metas_temp));
  CUDA_CHECK(cudaFreeHost(h_vectors_temp));

  CUDA_CHECK(cudaFree(d_keys_temp));
  CUDA_CHECK(cudaFree(d_metas_temp));
  CUDA_CHECK(cudaFree(d_vectors_temp));

  CUDA_CHECK(cudaDeviceSynchronize());

  CudaCheckError();
}

//TEST(MerlinHashTableTest, test_basic) {
//  test_basic(16, true);
//  test_basic(0, true);
//  test_basic(16, false);
//  test_basic(0, false);
//}
//TEST(MerlinHashTableTest, test_basic_when_full) {
//  test_basic_when_full(16, true);
//  test_basic_when_full(0, true);
//  test_basic_when_full(16, false);
//  test_basic_when_full(0, false);
//}
//TEST(MerlinHashTableTest, test_erase_if_pred) {
//  test_erase_if_pred(16, true);
//  test_erase_if_pred(0, true);
//  test_erase_if_pred(16, false);
//  test_erase_if_pred(0, false);
//}
TEST(MerlinHashTableTest, test_rehash) {
  test_rehash(16, true);
  test_rehash(0, true);
  test_rehash(16, false);
  test_rehash(0, false);
}
//TEST(MerlinHashTableTest, test_rehash_on_big_batch) {
//  test_rehash_on_big_batch(16, true);
//  test_rehash_on_big_batch(0, true);
//  test_rehash_on_big_batch(16, false);
//  test_rehash_on_big_batch(0, false);
//}
//TEST(MerlinHashTableTest, test_dynamic_rehash_on_multi_threads) {
//  test_dynamic_rehash_on_multi_threads(16, true);
//  test_dynamic_rehash_on_multi_threads(0, true);
//  test_dynamic_rehash_on_multi_threads(16, false);
//  test_dynamic_rehash_on_multi_threads(0, false);
//}
//TEST(MerlinHashTableTest, test_export_batch_if) {
//  test_export_batch_if(16, true);
//  test_export_batch_if(0, true);
//  test_export_batch_if(16, false);
//  test_export_batch_if(0, false);
//}
//TEST(MerlinHashTableTest, test_basic_for_cpu_io) {
//  test_basic_for_cpu_io(true);
//  test_basic_for_cpu_io(false);
//}
//
//TEST(MerlinHashTableTest, test_evict_strategy_lru_basic) {
//  test_evict_strategy_lru_basic(16, true);
//  test_evict_strategy_lru_basic(0, true);
//  test_evict_strategy_lru_basic(16, false);
//  test_evict_strategy_lru_basic(0, false);
//}
//
//TEST(MerlinHashTableTest, test_evict_strategy_customized_basic) {
//  test_evict_strategy_customized_basic(16, true);
//  test_evict_strategy_customized_basic(0, true);
//  test_evict_strategy_customized_basic(16, false);
//  test_evict_strategy_customized_basic(0, false);
//}
//
//TEST(MerlinHashTableTest, test_evict_strategy_customized_advanced) {
//  test_evict_strategy_customized_advanced(16, true);
//  test_evict_strategy_customized_advanced(0, true);
//  test_evict_strategy_customized_advanced(16, false);
//  test_evict_strategy_customized_advanced(0, false);
//}
//
//TEST(MerlinHashTableTest, test_evict_strategy_customized_correct_rate) {
//  test_evict_strategy_customized_correct_rate(16, true);
//  test_evict_strategy_customized_correct_rate(0, true);
//  test_evict_strategy_customized_correct_rate(16, false);
//  test_evict_strategy_customized_correct_rate(0, false);
//}
