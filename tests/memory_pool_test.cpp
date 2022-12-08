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

#include "merlin/memory_pool.cuh"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <iostream>

void print_divider() {
  for (size_t i = 0; i < 80; ++i) std::cout << '-';
  std::cout << std::endl;
}

void print_pool_options(const MemoryPoolOptions& opt) {
  print_divider();
  std::cout << "Memory Pool Configuration" << std::endl;
  print_divider();
  std::cout << "opt.buffer_size : " << opt.buffer_size << " bytes" << std::endl;
  std::cout << "opt.max_stock : " << opt.max_stock << " buffers" << std::endl;
  std::cout << "opt.base_stock : " << opt.base_stock << " buffers" << std::endl;
  std::cout << "opt.max_pending : " << opt.max_pending << " buffers "
            << std::endl;
  print_divider();
}

MemoryPoolOptions opt{
    4096,  //< buffer_size
    3,     //< max_stock
    2,     //< base_stock
    5,     //< max_pending
};

struct SomeType {
  int a;
  float b;

  friend std::ostream& operator<<(std::ostream&, const SomeType&);
};

std::ostream& operator<<(std::ostream& os, const SomeType& obj) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, &obj));

  SomeType tmp;
  if (attr.type == cudaMemoryTypeDevice) {
    CUDA_CHECK(
        cudaMemcpy(&tmp, &obj, sizeof(SomeType), cudaMemcpyDeviceToHost));
  } else {
    tmp = obj;
  }

  os << "a = " << tmp.a << ", b = " << tmp.b;
  return os;
}

void test_standard_allocator() {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;

  {
    auto ptr = Allocator::make_unique(1);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, nullptr);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_host_allocator() {
  using Allocator = DebugAllocator<HostAllocator<SomeType>>;

  {
    auto ptr = Allocator::make_unique(1);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, nullptr);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    ptr->a = 47;
    ptr->b = 11;
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }
}

void test_device_allocator() {
  using Allocator = DebugAllocator<DeviceAllocator<SomeType>>;

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  MERLIN_CHECK(num_devices > 0,
               "Need at least one CUDA capable device for running this test.");

  CUDA_CHECK(cudaSetDevice(num_devices - 1));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  {
    auto ptr = Allocator::make_unique(1);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Sync UPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "Sync UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_unique(1, stream);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "Async UPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "Async UPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  {
    auto ptr = Allocator::make_shared(1, stream);
    ASSERT_NE(ptr.get(), nullptr);

    std::cout << "SPtr after alloc: " << *ptr << std::endl;
    const SomeType tmp{47, 11};
    CUDA_CHECK(
        cudaMemcpy(ptr.get(), &tmp, sizeof(SomeType), cudaMemcpyHostToDevice));
    std::cout << "SPtr after set: " << *ptr << std::endl;

    ptr.reset();
    ASSERT_EQ(ptr.get(), nullptr);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_deplete_replenish() {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;

  // print_pool_options(opt);
  MemoryPool<Allocator> pool(opt);

  // Should have base_stock available after startup.
  std::cout << ".:: Initial state ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Should have no stock left after deplete.
  pool.deplete();
  std::cout << ".:: Deplete ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), 0);

  // After replenish initial state should be restored.
  std::cout << ".:: Replenish ::." << std::endl << pool << std::endl;
  pool.replenish();
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
}

void test_borrow_return_no_context() {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;

  // print_pool_options(opt);
  MemoryPool<Allocator> pool(opt);

  // Initial status.
  std::cout << ".:: Initial state ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow one buffer.
  {
    auto buffer = pool.get_unique();
    std::cout << ".:: Borrow 1 ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.base_stock - 1);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return one buffer.
  std::cout << ".:: Return 1 ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow workspace.
  {
    auto ws = pool.get_workspace<2>();
    std::cout << ".:: Borrow 2 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.base_stock - 2);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return workspace.
  std::cout << ".:: Return 2 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow dynamic workspace.
  {
    auto ws = pool.get_workspace(opt.base_stock);
    std::cout << ".:: Borrow 2 (dynamic) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  std::cout << ".:: Return 2 (dynamic) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow workspace that exceeds base pool size.
  {
    auto ws = pool.get_workspace<3>();
    std::cout << ".:: Borrow 3 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return workspace.
  std::cout << ".:: Return 3 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.max_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow workspace that exceeds maximum pool size.
  {
    auto ws = pool.get_workspace<4>();
    std::cout << ".:: Borrow 4 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return workspace.
  std::cout << ".:: Return 4 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.max_stock);
  ASSERT_EQ(pool.num_pending(), 0);
}

void test_borrow_return_with_context() {
  using Allocator = DebugAllocator<StandardAllocator<SomeType>>;

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  MERLIN_CHECK(num_devices > 0,
               "Need at least one CUDA capable device for running this test.");

  CUDA_CHECK(cudaSetDevice(num_devices - 1));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // print_pool_options(opt);
  MemoryPool<Allocator> pool(opt);

  // Initial status.
  std::cout << ".:: Initial state ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow/return one buffer.
  {
    auto buffer = pool.get_unique(stream);
    std::cout << ".:: Borrow 1 ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.base_stock - 1);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  std::cout << ".:: Return 1 ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock - 1);
  ASSERT_EQ(pool.num_pending(), 1);

  pool.await_pending();
  std::cout << ".:: Await pending ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow/return a workspace.
  {
    auto ws = pool.get_workspace<2>(stream);
    std::cout << ".:: Borrow 2 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.base_stock - 2);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  std::cout << ".:: Return 2 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock - 2);
  ASSERT_EQ(pool.num_pending(), 2);

  pool.await_pending();
  std::cout << ".:: Await pending ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow dynamic workspace.
  {
    auto ws = pool.get_workspace(opt.base_stock, stream);
    std::cout << ".:: Borrow 2 (dynamic) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), opt.base_stock - 2);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  std::cout << ".:: Return 2 (dynamic) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock - 2);
  ASSERT_EQ(pool.num_pending(), 2);

  pool.await_pending();
  std::cout << ".:: Await pending ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.base_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow workspace that exceeds base pool size.
  {
    auto ws = pool.get_workspace<3>(stream);
    std::cout << ".:: Borrow 3 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return workspace.
  std::cout << ".:: Return 3 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), 0);
  ASSERT_EQ(pool.num_pending(), 3);

  pool.await_pending();
  std::cout << ".:: Await pending ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.max_stock);
  ASSERT_EQ(pool.num_pending(), 0);

  // Borrow workspace that exceeds maximum pool size.
  {
    auto ws = pool.get_workspace<6>(stream);
    std::cout << ".:: Borrow 6 (static) ::." << std::endl << pool << std::endl;
    ASSERT_EQ(pool.current_stock(), 0);
    ASSERT_EQ(pool.num_pending(), 0);
  }

  // Return workspace.
  std::cout << ".:: Return 6 (static) ::." << std::endl << pool << std::endl;
  ASSERT_EQ(pool.current_stock(), opt.max_stock);
  ASSERT_EQ(pool.num_pending(), 1);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(MemoryPool, standard_allocator) { test_standard_allocator(); }
TEST(MemoryPool, host_allocator) { test_host_allocator(); }
TEST(MemoryPool, device_allocator) { test_device_allocator(); }

TEST(MemoryPool, deplete_replenish) { test_deplete_replenish(); }
TEST(MemoryPool, borrow_return_no_context) { test_borrow_return_no_context(); }

TEST(MemoryPool, borrow_return_with_context) {
  test_borrow_return_with_context();
}
