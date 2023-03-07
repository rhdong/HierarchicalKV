#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

using ValueType = int;
constexpr size_t interval = 16;

constexpr size_t num_buckets = 1024 * 1024;
constexpr size_t num_values_per_bucket = 2048;
constexpr size_t num_values = num_buckets * num_values_per_bucket;

constexpr size_t memory_pool_size =
    num_values * sizeof(ValueType);  // = 1024 * 1024 * 2048 * 4 = 8GB
constexpr size_t bucket_size = num_values_per_bucket * sizeof(ValueType);

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    std::ostringstream os;
    os << file << ':' << line << ": CUDA error " << cudaGetErrorName(val)
       << " (#" << val << "): " << cudaGetErrorString(val);
    throw CudaException(os.str());
  }
}

#define CUDA_CHECK(val)                     \
  do {                                      \
    cuda_check_((val), __FILE__, __LINE__); \
  } while (0)

struct __align__(16) Bucket {
  uint64_t* a;        // ignore it!
  uint64_t* b;        // ignore it!
  uint64_t* c;        // ignore it!
  ValueType* values;  // <<<----important member
  uint64_t d;         // ignore it!
  uint64_t e;         // ignore it!
  int f;              // ignore it!
};

__global__ void write_read(Bucket* buckets, int bucket_idx) {
  int value_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  ValueType* values = buckets[bucket_idx].values;
  *(values + value_idx * interval) =
      bucket_idx * (num_values_per_bucket / interval) + value_idx;
}

__global__ void read_when_error(Bucket* buckets, int bucket_idx,
                                int value_idx) {
  ValueType* values = buckets[bucket_idx].values;
  ValueType val = *(values + value_idx * interval);
  printf("[device side] ptr=%p\tval=%d\n", (values + value_idx * interval),
         val);
}

int main() {
  Bucket* buckets;

  // Allocating the buckets structs.
  cudaMallocManaged(&buckets, sizeof(Bucket) * num_buckets);

  std::cout << "size of Bucket=" << sizeof(Bucket) << std::endl;

  assert(num_buckets == (1024 * 1024));
  assert(memory_pool_size == (8ul << 30));
  assert(bucket_size == (128 * 4 * 16));
  assert(memory_pool_size == (bucket_size * num_buckets));

  // Allocating a memory pool on host memory for all of the values.
  ValueType* host_memory_pool;
  CUDA_CHECK(cudaHostAlloc(&host_memory_pool, memory_pool_size,
                           cudaHostAllocMapped | cudaHostAllocWriteCombined));

  // Make the `values` pointer of each bucket point to different section of the
  // memory pool.
  for (int i = 0; i < num_buckets; i++) {
    ValueType* h_memory_pool_section =
        host_memory_pool + (num_values_per_bucket * i);
    CUDA_CHECK(cudaHostGetDevicePointer(&(buckets[i].values),
                                        h_memory_pool_section, 0));
  }
  std::cout << "[finish allocating]"
            << "num_buckets=" << num_buckets
            << ", memory_pool_size=" << (8ul << 30)
            << ", bucket_size=" << (128 * 4 * 16) << std::endl;

  // Writing a `value` into `values` of each buckets every `interval` values.
  // `value = bucket_idx * (num_values_per_bucket / interval) + value_idx`.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  for (int i = 0; i < num_buckets; i++) {
    assert(128 == (num_values_per_bucket / interval));
    write_read<<<1, (num_values_per_bucket / interval), 0, stream>>>(buckets,
                                                                     i);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "[finish writing]" << std::endl;

  // Checking if the value is expected from both sides of device and host.
  size_t error_num = 0;
  size_t correct_num = 0;
  for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
    for (int value_idx = 0; value_idx < (num_values_per_bucket / interval);
         value_idx++) {
      ValueType expected_value =
          bucket_idx * (num_values_per_bucket / interval) + value_idx;
      ValueType* h_ptr = host_memory_pool + bucket_idx * num_values_per_bucket +
                         value_idx * interval;
      ValueType host_value = *h_ptr;
      if (host_value != expected_value && host_value != 0) {
        read_when_error<<<1, 1, 0, stream>>>(buckets, bucket_idx, value_idx);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("[host   side]: ptr=%p\tval=%d\n\n", h_ptr, host_value);
        error_num++;
      } else {
        correct_num++;
      }
    }
  }
  std::cout << "error_num=" << error_num << "\tcorrect_num=" << correct_num
            << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "[finish checking]" << std::endl;

  CUDA_CHECK(cudaFreeHost(host_memory_pool));
  CUDA_CHECK(cudaFree(buckets));
}