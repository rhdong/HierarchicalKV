/*
 * Review-response correctness stress gate.
 *
 * These tests document and verify the legal API contract most likely to be
 * challenged by reviewers: duplicate-key handling, dual-bucket unique-key
 * requirements, repeated-batch updates, and score-based admission behavior.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "merlin_hashtable.cuh"
#include "test_util.cuh"

namespace {

constexpr size_t kDim = 16;
using K = uint64_t;
using V = float;
using S = uint64_t;
using TableOptions = nv::merlin::HashTableOptions;
using TableMode = nv::merlin::TableMode;
using EvictStrategy = nv::merlin::EvictStrategy;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

void create_table(Table& table, TableMode mode, size_t capacity,
                  size_t dim = kDim) {
  TableOptions options;
  options.init_capacity = capacity;
  options.max_capacity = capacity;
  options.max_hbm_for_vectors = 0;
  options.dim = dim;
  options.max_bucket_size = 128;
  options.table_mode = mode;
  table.init(options);
}

void fill_sentinel_values(std::vector<V>& values, size_t n, size_t dim,
                          V sentinel) {
  values.assign(n * dim, sentinel);
}

template <typename T>
T* copy_to_device(const std::vector<T>& host) {
  T* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}

size_t count_found(const std::vector<bool>& found) {
  return std::count(found.begin(), found.end(), true);
}

std::vector<bool> find_keys(Table& table, const std::vector<K>& keys,
                            std::vector<V>* values_out = nullptr) {
  K* d_keys = copy_to_device(keys);
  V* d_values = nullptr;
  bool* d_found = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, keys.size() * kDim * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_found, keys.size() * sizeof(bool)));

  table.find(keys.size(), d_keys, d_values, d_found, nullptr, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint8_t> found_bytes(keys.size());
  CUDA_CHECK(cudaMemcpy(found_bytes.data(), d_found,
                        keys.size() * sizeof(bool), cudaMemcpyDeviceToHost));
  std::vector<bool> found(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    found[i] = found_bytes[i] != 0;
  }

  if (values_out != nullptr) {
    values_out->resize(keys.size() * kDim);
    CUDA_CHECK(cudaMemcpy(values_out->data(), d_values,
                          keys.size() * kDim * sizeof(V),
                          cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_found));
  return found;
}

void insert_or_assign(Table& table, const std::vector<K>& keys,
                      const std::vector<V>& values,
                      const std::vector<S>& scores, bool unique_key,
                      bool ignore_evict_strategy = false) {
  K* d_keys = copy_to_device(keys);
  V* d_values = copy_to_device(values);
  S* d_scores = copy_to_device(scores);

  table.insert_or_assign(keys.size(), d_keys, d_values, d_scores, 0,
                         unique_key, ignore_evict_strategy);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
}

bool is_whole_sentinel_vector(const std::vector<V>& values, size_t row,
                              const std::vector<V>& legal_sentinels) {
  const V first = values[row * kDim];
  if (std::find(legal_sentinels.begin(), legal_sentinels.end(), first) ==
      legal_sentinels.end()) {
    return false;
  }
  for (size_t d = 1; d < kDim; d++) {
    if (values[row * kDim + d] != first) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(CorrectnessStress, DuplicateSameKeyBatchThroughputNoTornValue) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 1024;
  constexpr K kHotKey = 42;

  Table table;
  create_table(table, TableMode::kThroughput, kCapacity);

  std::vector<K> keys(kBatch, kHotKey);
  std::vector<V> values(kBatch * kDim);
  std::vector<S> scores(kBatch);
  std::vector<V> legal_sentinels(kBatch);

  for (size_t i = 0; i < kBatch; i++) {
    const V sentinel = static_cast<V>(1000 + i);
    legal_sentinels[i] = sentinel;
    scores[i] = i + 1;
    for (size_t d = 0; d < kDim; d++) {
      values[i * kDim + d] = sentinel;
    }
  }

  insert_or_assign(table, keys, values, scores, false);

  EXPECT_EQ(table.size(0), static_cast<size_t>(1));

  std::vector<V> found_values;
  auto found = find_keys(table, std::vector<K>{kHotKey}, &found_values);
  ASSERT_EQ(count_found(found), static_cast<size_t>(1));
  EXPECT_TRUE(is_whole_sentinel_vector(found_values, 0, legal_sentinels))
      << "duplicate same-key update produced a torn or unknown value";
}

TEST(CorrectnessStress, DualBucketRejectsNonUniqueBatchContract) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 16;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> keys(kBatch, 7);
  std::vector<V> values(kBatch * kDim, 1.0f);
  std::vector<S> scores(kBatch, 1);

  K* d_keys = copy_to_device(keys);
  V* d_values = copy_to_device(values);
  S* d_scores = copy_to_device(scores);

  EXPECT_THROW(table.insert_or_assign(kBatch, d_keys, d_values, d_scores, 0,
                                      false),
               std::runtime_error);

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_scores));
}

TEST(CorrectnessStress, DualBucketRepeatedBatchUpdateNoDuplicates) {
  constexpr size_t kCapacity = 128 * 1024;
  constexpr size_t kBatch = 2048;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> keys(kBatch);
  std::iota(keys.begin(), keys.end(), 1);
  std::vector<S> scores(kBatch);
  std::iota(scores.begin(), scores.end(), 1);

  std::vector<V> values_v1;
  std::vector<V> values_v2;
  fill_sentinel_values(values_v1, kBatch, kDim, 1.0f);
  fill_sentinel_values(values_v2, kBatch, kDim, 2.0f);

  insert_or_assign(table, keys, values_v1, scores, true);
  insert_or_assign(table, keys, values_v2, scores, true);

  EXPECT_EQ(table.size(0), kBatch);

  std::vector<V> found_values;
  auto found = find_keys(table, keys, &found_values);
  ASSERT_EQ(count_found(found), kBatch);
  for (size_t i = 0; i < kBatch; i++) {
    for (size_t d = 0; d < kDim; d++) {
      EXPECT_FLOAT_EQ(found_values[i * kDim + d], 2.0f)
          << "key " << keys[i] << " dim " << d;
    }
  }
}

TEST(CorrectnessStress, MemoryModeScoreAdmissionAccounting) {
  constexpr size_t kCapacity = 128 * 128;
  constexpr size_t kFill = kCapacity;
  constexpr size_t kBurst = 512;

  Table table;
  create_table(table, TableMode::kMemory, kCapacity);

  std::vector<K> fill_keys(kFill);
  std::iota(fill_keys.begin(), fill_keys.end(), 1);
  std::vector<V> fill_values(kFill * kDim, 1.0f);
  std::vector<S> fill_scores(kFill);
  for (size_t i = 0; i < kFill; i++) {
    fill_scores[i] = 1000000 + i;
  }

  insert_or_assign(table, fill_keys, fill_values, fill_scores, true, true);
  const size_t size_after_fill = table.size(0);
  ASSERT_GT(size_after_fill, static_cast<size_t>(kFill * 0.95))
      << "admission accounting test requires a near-full table";

  std::vector<K> low_keys(kBurst);
  std::iota(low_keys.begin(), low_keys.end(), kFill + 1);
  std::vector<V> low_values(kBurst * kDim, 2.0f);
  std::vector<S> low_scores(kBurst, 1);

  insert_or_assign(table, low_keys, low_values, low_scores, true, false);
  const size_t size_after_low = table.size(0);
  EXPECT_LE(size_after_low, kCapacity);

  auto low_found = find_keys(table, low_keys);
  const size_t low_accepted = count_found(low_found);

  std::vector<K> high_keys(kBurst);
  std::iota(high_keys.begin(), high_keys.end(), kFill + 1 + kBurst);
  std::vector<V> high_values(kBurst * kDim, 3.0f);
  std::vector<S> high_scores(kBurst, 1000000000);

  insert_or_assign(table, high_keys, high_values, high_scores, true, false);
  EXPECT_LE(table.size(0), kCapacity);

  auto high_found = find_keys(table, high_keys);
  const size_t high_accepted = count_found(high_found);

  EXPECT_LT(low_accepted, kBurst)
      << "low-score burst should not be fully admitted near capacity";
  EXPECT_GT(high_accepted, low_accepted)
      << "high-score burst should admit more keys than low-score burst";
}
