/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#pragma once

#include <cstddef>
#include <cstdint>
#include "merlin/core_kernels/kernel_utils.cuh"
#include "merlin_hashtable.cuh"
#include "merlin_hashtable_base.hpp"

namespace nv {
namespace merlin {
namespace device {

template <typename K, typename V, typename S>
struct HashTableDeviceView {
  using base_type = HashTableBase<K, V, S>;
  using key_type = typename base_type::key_type;
  using value_type = typename base_type::value_type;
  using score_type = typename base_type::score_type;
  using bucket_type = typename base_type::bucket_type;

  // Read-only view. Non-const to match find_without_lock_readonly_no_sync.
  bucket_type* buckets;
  size_t bucket_count;
  size_t bucket_max_size;
};

// Read-only contract:
// - No concurrent writes while lookups are running.
// - Keys in the same lookup batch must be non-overlapping.
template <typename K, typename V, typename S, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ int readonly_lookup_no_sync(
    const HashTableDeviceView<K, V, S>& view,
    const typename HashTableDeviceView<K, V, S>::key_type key,
    const uint32_t rank) {
  if (view.buckets == nullptr || view.bucket_count == 0 ||
      view.bucket_max_size == 0) {
    return -1;
  }
  const K hashed_key = Murmur3HashDevice(key);
  const size_t global_idx =
      hashed_key % (view.bucket_count * view.bucket_max_size);
  const uint32_t bucket_idx =
      static_cast<uint32_t>(global_idx / view.bucket_max_size);
  const uint32_t start_idx =
      static_cast<uint32_t>(global_idx % view.bucket_max_size);
  const uint32_t aligned_start = start_idx - (start_idx % TILE_SIZE);
  return find_without_lock_readonly_no_sync<K, V, S, TILE_SIZE>(
      view.buckets + bucket_idx, key, aligned_start,
      static_cast<uint32_t>(view.bucket_max_size), rank);
}

template <typename K, typename V, typename S, int Strategy, typename ArchTag>
__host__ inline HashTableDeviceView<K, V, S> make_device_view(
    const HashTable<K, V, S, Strategy, ArchTag>& table) {
  HashTableDeviceView<K, V, S> view{};
  view.buckets = table.device_buckets();
  view.bucket_count = table.device_bucket_count();
  view.bucket_max_size = table.device_bucket_max_size();
  return view;
}

}  // namespace device
}  // namespace merlin
}  // namespace nv
