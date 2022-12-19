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

#pragma once

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "debug.hpp"

namespace nv {
namespace merlin {

/**
 * Allocators are used by the memory pool (and maybe other classes) to create
 * RAII complient containers for buffers allocated in different memory areas.
 */
template <class T, class Allocator>
struct AllocatorBase {
  using type = T;
  using sync_unique_ptr = std::unique_ptr<type, Allocator>;
  using async_unique_ptr = std::unique_ptr<type, std::function<void(type*)>>;
  using shared_ptr = std::shared_ptr<type>;

  inline static sync_unique_ptr make_unique(size_t n) {
    return sync_unique_ptr(Allocator::alloc(n));
  }

  inline static async_unique_ptr make_unique(size_t n, cudaStream_t stream) {
    return {Allocator::alloc(n, stream),
            [stream](type* p) { Allocator::free(p); }};
  }

  inline static shared_ptr make_shared(size_t n, cudaStream_t stream = 0) {
    return {Allocator::alloc(n, stream),
            [stream](type* p) { Allocator::free(p, stream); }};
  }

  inline void operator()(type* ptr) { Allocator::free(ptr); }
};

/**
 * Trivial fallback implementation using the standard C++ allocator. This mostly
 * exists to ensure interface correctness, and as an illustration of what a
 * proper allocator implementation should look like.
 */
template <class T>
struct StandardAllocator final : AllocatorBase<T, StandardAllocator<T>> {
  using type = typename AllocatorBase<T, StandardAllocator<T>>::type;

  static constexpr const char* name{"StandardAllocator"};

  inline static type* alloc(size_t n, cudaStream_t stream = 0) {
    return new type[n];
  }

  inline static void free(type* ptr, cudaStream_t stream = 0) { delete[] ptr; }
};

/**
 * Claim/release buffers in pinned host memory.
 */
template <class T>
struct HostAllocator final : AllocatorBase<T, HostAllocator<T>> {
  using type = typename AllocatorBase<T, HostAllocator<T>>::type;

  static constexpr const char* name{"HostAllocator"};

  inline static type* alloc(size_t n, cudaStream_t stream = 0) {
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, sizeof(T) * n));
    return reinterpret_cast<type*>(ptr);
  }

  inline static void free(type* ptr, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};

/**
 * Claim/release buffers in the active CUDA device. Will not test if the correct
 * device was used, and throw if CUDA runtime API response is negative.
 */
template <class T>
struct DeviceAllocator final : AllocatorBase<T, DeviceAllocator<T>> {
  using type = typename AllocatorBase<T, DeviceAllocator<T>>::type;

  static constexpr const char* name{"DeviceAllocator"};

  inline static type* alloc(size_t n, cudaStream_t stream = 0) {
    void* ptr;
    cudaError_t res;
    if (stream) {
      res = cudaMallocAsync(&ptr, sizeof(T) * n, stream);
    } else {
      res = cudaMalloc(&ptr, sizeof(T) * n);
    }
    CUDA_CHECK(res);
    return reinterpret_cast<type*>(ptr);
  }

  inline static void free(type* ptr, cudaStream_t stream = 0) {
    cudaError_t res;
    if (stream) {
      res = cudaFreeAsync(ptr, stream);
    } else {
      res = cudaFree(ptr);
    }
    CUDA_CHECK(res);
  }
};

/**
 * Helper structure to configure a memory pool.
 */
struct MemoryPoolOptions {
  size_t buffer_size =
      256L * 1024 * 1024;  ///< Size in bytes for buffers produced by this pool.
  size_t max_stock = 4;    ///< Amount of buffers to keep in reserve.
  size_t base_stock = 3;   ///< Amount of buffers to create right away.
  size_t max_pending = 16;  ///< Maximum amount of awaitable buffers. If this
                            ///< limit is exceeded threads will start to block.
};

/**
 * Forward declares required to make templated ostream overload work.
 */
template <class Allocator>
class MemoryPool;

template <class Allocator>
std::ostream& operator<<(std::ostream&, const MemoryPool<Allocator>&);

/**
 * CUDA deferred execution aware memory pool implementation. As for every memory
 * pool, the general idea is to have resuable buffers. All buffers have the same
 * size.
 *
 * General behavior:
 *
 * This memory pool implementation attempts to avoid blocking before the fact,
 * but also avoids relying on a background worker.
 *
 * Buffer borrow and return semantics tightly align with C++ RAII principles.
 * That is, if a workspace is requested, any borrowed buffers will be returned
 * automatically when leaving the scope.
 *
 * You can either borrow a single buffer, or a workspace (that is multiple
 * buffers). We support dynamic and static workspaces. Static workspaces have
 * the benefit that they will never require heap memory (no hidden allocations).
 *
 *
 * Buffer borrowing:
 *
 * If buffers are requested, we take them from the stock, if available. If the
 * stock is depleted, we check if any pending buffer has been used up by the GPU
 * and adds them to the stock. If was also not successful, we allocate a new
 * buffer. Buffers or workspaces (groups of buffers).
 *
 * When borrowing a buffer a streaming context can be specified. This context is
 * relevant for allocation and during returns. It is assumed that the stream you
 * provide as context will be the stream where you queue the workload. Not doing
 * so may lead to undefined behavior.
 *
 * Buffer return:
 *
 * If no context is provided, we cannot make any assumptions regarding the usage
 * one the device. So we sychronize the device first and then return the buffer
 * to the stock. If a streaming context was provided, we queue an event and add
 * the buffer to the `pending` pool. That means, the buffer has been
 * reqlinquished by the CPU, but may still be used by the GPU. If no pending
 * slot is available, we probe the currently pending buffers events for
 * completion. Completed pending buffers are returned to the reserve. If so, we
 * queue the buffer in the freed slot. If that was unsucessful (i.e., all
 * currently pending buffers are still in use by the GPU), we have no choice but
 * the free the buffer using the current stream.
 *
 * In either case, `max_reserve` represents the maxmum size of the stock. If
 * returning a buffer would lead to the stock exeeding this quantity, the buffer
 * is queued for destruction.
 */
template <class Allocator>
class MemoryPool final {
 public:
  using pool_type = MemoryPool<Allocator>;
  using alloc_type = typename Allocator::type;
  template <class Container>
  class Workspace {
   public:
    inline Workspace() : pool_{nullptr}, stream_{0} {}

    inline Workspace(pool_type* pool, cudaStream_t stream)
        : pool_{pool}, stream_{stream} {}

    Workspace(const Workspace&) = delete;

    Workspace& operator=(const Workspace&) = delete;

    inline Workspace(Workspace&& other)
        : pool_{other.pool_},
          stream_{other.stream_},
          buffers_{std::move(other.buffers_)} {}

    inline Workspace& operator=(Workspace&& other) {
      if (pool_) {
        pool_->put_raw(buffers_.begin(), buffers_.end(), stream_);
      }
      pool_ = other.pool_;
      stream_ = other.stream_;
      buffers_ = std::move(other.buffers_);
      other.pool_ = nullptr;
      return *this;
    }

    inline ~Workspace() {
      if (pool_) {
        pool_->put_raw(buffers_.begin(), buffers_.end(), stream_);
      }
    }

    template <class T>
    constexpr void at(const size_t n, T* ptr) const {
      *ptr = at<T>(n);
    }

    template <class T>
    constexpr T at(const size_t n) const {
      return reinterpret_cast<T>(buffers_.at(n));
    }

    template <class T>
    constexpr void get(const size_t n, T* ptr) const {
      *ptr = get<T>(n);
    }

    template <class T>
    constexpr T get(const size_t n) const {
      return reinterpret_cast<T>(buffers_[n]);
    }

    constexpr alloc_type* operator[](const size_t n) const {
      return buffers_[n];
    }

   protected:
    pool_type* pool_;
    cudaStream_t stream_;
    Container buffers_;
  };

  template <size_t N>
  class StaticWorkspace final : public Workspace<std::array<alloc_type*, N>> {
   public:
    using base_type = Workspace<std::array<alloc_type*, N>>;

    friend class MemoryPool<Allocator>;

    inline StaticWorkspace() : base_type() {}

    StaticWorkspace(const StaticWorkspace&) = delete;

    StaticWorkspace& operator=(const StaticWorkspace&) = delete;

    inline StaticWorkspace(StaticWorkspace&& other)
        : base_type(std::move(other)) {}

    inline StaticWorkspace& operator=(StaticWorkspace&& other) {
      base_type::operator=(std::move(other));
      return *this;
    }

   private:
    inline StaticWorkspace(pool_type* pool, cudaStream_t stream)
        : base_type(pool, stream) {
      auto& buffers = this->buffers_;
      pool->get_raw(buffers.begin(), buffers.end(), stream);
    }
  };

  class DynamicWorkspace final : public Workspace<std::vector<alloc_type*>> {
   public:
    using base_type = Workspace<std::vector<alloc_type*>>;

    friend class MemoryPool<Allocator>;

    inline DynamicWorkspace() : base_type() {}

    DynamicWorkspace(const DynamicWorkspace&) = delete;

    DynamicWorkspace& operator=(const DynamicWorkspace&) = delete;

    inline DynamicWorkspace(DynamicWorkspace&& other)
        : base_type(std::move(other)) {}

    inline DynamicWorkspace& operator=(DynamicWorkspace&& other) {
      base_type::operator=(std::move(other));
      return *this;
    }

   private:
    inline DynamicWorkspace(pool_type* pool, size_t n, cudaStream_t stream)
        : base_type(pool, stream) {
      auto& buffers = this->buffers_;
      buffers.resize(n);
      pool->get_raw(buffers.begin(), buffers.end(), stream);
    }
  };

  MemoryPool(const MemoryPoolOptions& options) : options_{options} {
    MERLIN_CHECK(options_.buffer_size > 0,
                 "[HierarchicalKV] Memory pool buffer size invalid!");
    MERLIN_CHECK(
        options_.base_stock <= options_.max_stock,
        "[hierarchicalKV] Initial reserve cannot exceed maximum reserve!");

    // Create initial buffer stock.
    stock_.reserve(options_.max_stock);
    while (stock_.size() < options_.base_stock) {
      stock_.emplace_back(Allocator::alloc(options_.buffer_size));
    }

    // Create enough events, so we have one per potentially pending buffer.
    ready_events_.resize(options_.max_pending);
    for (auto& ready_event : ready_events_) {
      CUDA_CHECK(cudaEventCreate(&ready_event));
    }

    // Preallocate pending.
    pending_.reserve(options_.max_pending);
  }

  ~MemoryPool() {
    // Make sure all queued tasks are complete.
    await_pending();

    // Free event and buffer memory.
    for (auto& ready_event : ready_events_) {
      CUDA_CHECK(cudaEventDestroy(ready_event));
    }

    for (auto& ptr : stock_) {
      Allocator::free(ptr);
    }
  }

  inline size_t buffer_size() const { return options_.buffer_size; }

  inline size_t max_batch_size(size_t max_item_size) const {
    return options_.buffer_size / max_item_size;
  }

  template <class T>
  inline size_t max_batch_size() const {
    return max_batch_size(sizeof(T));
  }

  size_t current_stock() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stock_.size();
  }

  size_t num_pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
  }

  void deplete(cudaStream_t stream = 0) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Collect any pending buffers first.
    collect_pending_unsafe(stream);

    // Deplete remaining buffers.
    for (auto& ptr : stock_) {
      Allocator::free(ptr, stream);
    }
    stock_.clear();
  }

  void replenish(cudaStream_t stream = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stock_.size() >= options_.base_stock) {
      return;
    }

    // Try to collect pending buffers first.
    collect_pending_unsafe(stream);

    // Fill up until we reach the base stock.
    while (stock_.size() < options_.base_stock) {
      stock_.emplace_back(Allocator::alloc(options_.buffer_size, stream));
    }

    // To avoid trouble downstream, make sure the buffers have materialized.
    if (stream) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  void await_pending(cudaStream_t stream = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!pending_.empty()) {
      collect_pending_unsafe(stream);
      if (pending_.empty()) {
        break;
      }
      std::this_thread::yield();
    }
  }

  inline std::unique_ptr<alloc_type, std::function<void(alloc_type*)>>
  get_unique(cudaStream_t stream = 0) {
    alloc_type* ptr;
    get_raw(&ptr, (&ptr) + 1, stream);
    return {ptr,
            [this, stream](alloc_type* p) { put_raw(&p, (&p) + 1, stream); }};
  }

  inline std::shared_ptr<alloc_type> get_shared(cudaStream_t stream = 0) {
    alloc_type* ptr;
    get_raw(&ptr, (&ptr) + 1, stream);
    return {ptr,
            [this, stream](alloc_type* p) { put_raw(&p, (&p) + 1, stream); }};
  }

  template <size_t N>
  inline StaticWorkspace<N> get_workspace(cudaStream_t stream = 0) {
    return {this, stream};
  }

  inline DynamicWorkspace get_workspace(size_t n, cudaStream_t stream = 0) {
    return {this, n, stream};
  }

  friend std::ostream& operator<<<Allocator>(std::ostream&, const MemoryPool&);

 private:
  inline void collect_pending_unsafe(cudaStream_t stream) {
    auto it = std::remove_if(
        pending_.begin(), pending_.end(),
        [this, stream](const std::pair<alloc_type*, cudaEvent_t>& pair) {
          const cudaError_t event_state = cudaEventQuery(pair.second);
          switch (event_state) {
            case cudaSuccess:
              // Stock buffers and destroy those that are no longer needed.
              if (stock_.size() < options_.max_stock) {
                stock_.emplace_back(pair.first);
              } else {
                Allocator::free(pair.first, stream);
              }
              ready_events_.emplace_back(pair.second);
              return true;
            case cudaErrorNotReady:
              return false;
            default:
              CUDA_CHECK(event_state);
              return false;
          }
        });
    pending_.erase(it, pending_.end());
  }

  template <class Iterator>
  inline void get_raw(Iterator first, Iterator const last,
                      cudaStream_t stream) {
    // Get pre-allocated buffers if stock available.
    {
      std::lock_guard<std::mutex> lock(mutex_);

      for (; first != last; ++first) {
        // If no buffers available, try to make some available.
        if (stock_.empty()) {
          collect_pending_unsafe(stream);
          if (stock_.empty()) {
            // No buffers available.
            break;
          }
        }

        // Just take the next available buffer.
        *first = stock_.back();
        stock_.pop_back();
      }
    }

    // Forge new buffers until request can be filled.
    for (; first != last; ++first) {
      *first = Allocator::alloc(options_.buffer_size, stream);
    }
  }

  template <class Iterator>
  inline void put_raw(Iterator first, Iterator const last,
                      cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);

    // If the workspace that borrowed a stream was moved out of the RAII scope
    // where it was created, it could happen that the stream was destroyed when
    // we return the buffer ownershup. This `cudaStreamQuery` will prevent that.
    if (stream && cudaStreamQuery(stream) != cudaErrorInvalidResourceHandle) {
      for (; first != last; ++first) {
        // Spin lock if too many pending buffers (i.e., let CPU wait for GPU).
        while (ready_events_.empty()) {
          collect_pending_unsafe(stream);
          if (!ready_events_.empty()) {
            break;
          }
          std::this_thread::yield();
        }

        // Queue buffer.
        cudaEvent_t ready_event = ready_events_.back();
        ready_events_.pop_back();
        CUDA_CHECK(cudaEventRecord(ready_event, stream));
        pending_.emplace_back(*first, ready_event);
      }
    } else {
      // Without stream context, we must force a hard sync with the GPU.
      CUDA_CHECK(cudaDeviceSynchronize());

      for (; first != last; ++first) {
        // Stock buffers and destroy those that are no longer needed.
        if (stock_.size() < options_.max_stock) {
          stock_.emplace_back(*first);
        } else {
          Allocator::free(*first);
        }
      }
    }
  }

  const MemoryPoolOptions options_;

  mutable std::mutex mutex_;
  std::vector<alloc_type*> stock_;
  std::vector<cudaEvent_t> ready_events_;
  std::vector<std::pair<alloc_type*, cudaEvent_t>> pending_;
};

template <class Allocator>
std::ostream& operator<<(std::ostream& os, const MemoryPool<Allocator>& pool) {
  for (size_t i = 0; i < 80; ++i) os << '-';
  os << std::endl << "Stock =" << std::endl;

  {
    std::lock_guard<std::mutex> lock(pool.mutex_);

    // Current stock.
    for (size_t i = 0; i < pool.stock_.size(); ++i) {
      os << "[ " << i << " ] " << static_cast<void*>(pool.stock_[i])
         << std::endl;
    }

    // Pending buffers.
    os << std::endl << "Pending =" << std::endl;
    for (size_t i = 0; i < pool.pending_.size(); ++i) {
      os << "[ " << i
         << " ] buffer = " << static_cast<void*>(pool.pending_[i].first)
         << ", ready_event = " << static_cast<void*>(pool.pending_[i].second)
         << std::endl;
    }

    // Available ready events.
    os << std::endl << "Ready Events =" << std::endl;
    for (size_t i = 0; i < pool.ready_events_.size(); ++i) {
      os << "[ " << i << " ] " << static_cast<void*>(pool.ready_events_[i])
         << std::endl;
    }
  }

  for (size_t i = 0; i < 80; ++i) os << '-';
  os << std::endl;

  return os;
}

}  // namespace merlin
}  // namespace nv
