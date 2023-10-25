// Copyright 2023 Vipshop Inc. All Rights Reserved.
// Author: Yifeng Zhan (ethan01.zhan@vipshop.com)
//

#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <experimental/filesystem>
#include <fstream>
#include "include/merlin/utils.cuh"
#include "include/merlin_hashtable.cuh"

namespace fs = std::experimental::filesystem;

template <typename T>
T StrTo(const std::string& s);

template <>
unsigned long StrTo<unsigned long>(const std::string& s) {
  return std::stoul(s);
}

int main(int argc, char** argv) {
  using K = uint64_t;
  using V = float;
  using S = uint64_t;
  constexpr int DIM = 16;

  using EvictStrategy = nv::merlin::EvictStrategy;
  using HashTableGpu = nv::merlin::HashTable<K, V, S, EvictStrategy::kEpochLru>;
  using TableOptions = nv::merlin::HashTableOptions;

  HashTableGpu table;
  TableOptions table_options;
  table_options.dim = DIM;
  table_options.init_capacity = 3145728;
  table_options.max_capacity = 3145728;
  table_options.max_hbm_for_vectors = 8096UL << 20;
  table.init(table_options);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  S epoch = 0l;

  for (const auto& entry : fs::directory_iterator(argv[1])) {
    size_t size = table.size(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("s, size = %zu\n", size);
    fs::path p = entry.path();
    std::ifstream f(p.string());
    epoch++;
    EvictStrategy::set_global_epoch(epoch);

    std::vector<K> h_keys;
    h_keys.reserve(1024 * 1024);
    std::string s;
    while (f >> s) {
      K key = StrTo<K>(s);
      h_keys.emplace_back(key);
    }

    thrust::device_vector<K> d_keys(h_keys);
    thrust::device_vector<V> d_values(h_keys.size() * DIM);
    const K* d_keys_ptr = thrust::raw_pointer_cast(d_keys.data());
    const V* d_values_ptr = thrust::raw_pointer_cast(d_values.data());
    table.insert_or_assign(h_keys.size(), d_keys_ptr, d_values_ptr,
                           nullptr /* score */, stream, epoch);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
