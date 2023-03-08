#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

using ValueType = void*;
constexpr int DIM = 8;

constexpr size_t num_vector_per_bucket = 128;
constexpr size_t num_buckets = 1024 * 1024;
constexpr size_t num_vector = num_buckets * num_vector_per_bucket;

constexpr size_t memory_pool_size =
    num_vector * sizeof(ValueType) * DIM;  // = 128 * 1024 * 1024 * 4 * 16 = 8GB

constexpr size_t bucket_vectors_size =
    num_vector_per_bucket * sizeof(ValueType) * DIM;

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
  uint64_t* a;         // ignore it!
  uint64_t* b;         // ignore it!
  ValueType* c;        // ignore it!
  ValueType* vectors;  // <<<----important member
  uint64_t d;          // ignore it!
  uint64_t e;          // ignore it!
  int f;               // ignore it!
};

__global__ void write_read(Bucket* buckets, int bucket_idx) {
  int vector_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  ValueType* vectors = buckets[bucket_idx].vectors;
  ValueType expected = static_cast<ValueType>(vectors + vector_idx * DIM);
  *(vectors + vector_idx * DIM) = expected;
}

__global__ void read_when_error(Bucket* buckets, int bucket_idx,
                                int vector_idx) {
  ValueType* vectors = buckets[bucket_idx].vectors;
  ValueType device_val = *(vectors + vector_idx * DIM);
  printf("device_val=%p\t", device_val);
}

__global__ void check_from_device(Bucket* buckets, int bucket_idx,
                                  int vector_idx, bool* correct) {
  ValueType* vectors = buckets[bucket_idx].vectors;
  ValueType device_val = *(vectors + vector_idx * DIM);
  ValueType expected = static_cast<ValueType>(vectors + vector_idx * DIM);
  *correct = (expected == device_val);
  if (!(*correct)) {
    printf("device side: ptr=%p\texpected=%p\tdevice_val=%p\t",
           (vectors + vector_idx * DIM), expected, device_val);
  }
}

int test() {
  Bucket* buckets;

  // Allocating the buckets structs.
  cudaMallocManaged(&buckets, sizeof(Bucket) * num_buckets);

  std::cout << "size of Bucket=" << sizeof(Bucket) << std::endl;

  assert(num_buckets == (1024 * 1024));
  assert(memory_pool_size == (8ul << 30));
  assert(bucket_vectors_size == (128 * 4 * 16));
  assert(memory_pool_size == (bucket_vectors_size * num_buckets));

  // Allocating a memory pool on host memory for all of the vectors.
  ValueType* host_memory_pool;
  CUDA_CHECK(cudaHostAlloc(&host_memory_pool, memory_pool_size,
                           cudaHostAllocMapped | cudaHostAllocWriteCombined));

  // Make the `vectors` pointer of each bucket point to different section of the
  // memory pool.
  for (int i = 0; i < num_buckets; i++) {
    ValueType* h_memory_pool =
        host_memory_pool + (num_vector_per_bucket * DIM * i);
    CUDA_CHECK(
        cudaHostGetDevicePointer(&(buckets[i].vectors), h_memory_pool, 0));
  }
  std::cout << "finish allocating"
            << ", num_buckets=" << num_buckets
            << ", bucket_vectors_size=" << (128 * 4 * 16)
            << ", memory_pool_size=" << (8ul << 30) << std::endl;

  // Writing a value into `values` of each buckets every `interval` values.
  // `value = the address of writing position`.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  for (int i = 0; i < num_buckets; i++) {
    write_read<<<1, num_vector_per_bucket, 0, stream>>>(buckets, i);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "finish writing" << std::endl;

  // Checking if the value is expected from both of device and host sides.
  size_t error_num = 0;
  size_t correct_num = 0;
  bool* d_correct;
  bool h_correct = false;
  CUDA_CHECK(cudaHostAlloc(&d_correct, sizeof(bool), cudaHostAllocMapped));
  for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
    for (int vector_idx = 0; vector_idx < num_vector_per_bucket; vector_idx++) {
      ValueType* host_ptr =
          (host_memory_pool + bucket_idx * num_vector_per_bucket * DIM +
           vector_idx * DIM + 0);  // write to first position of the `vector`.
      ValueType host_val = *host_ptr;

      // Check from device
      check_from_device<<<1, 1, 0, stream>>>(buckets, bucket_idx, vector_idx,
                                             d_correct);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (!(*d_correct)) {
        error_num++;
        printf("host_val=%p\n", host_val);
      }

      // Check from host
      ValueType expected = static_cast<ValueType>(host_ptr);
      h_correct = (host_val == expected);
      if (!h_correct) {
        printf("host   side: ptr=%p\texpected=%p\t", host_ptr, expected);
        read_when_error<<<1, 1, 0, stream>>>(buckets, bucket_idx, vector_idx);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("host_val=%p\n\n", host_val);
        error_num++;
      } else {
        correct_num++;
      }
    }
  }
  CUDA_CHECK(cudaFreeHost(d_correct));
  std::cout << "error_num=" << error_num << "\tcorrect_num=" << correct_num
            << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "finish checking" << std::endl;

  CUDA_CHECK(cudaFreeHost(host_memory_pool));
  CUDA_CHECK(cudaFree(buckets));
  return error_num;
}

int main() {
  int TEST_TIMES = 1;
  int fail_times = 0;
  for (int i = 0; i < TEST_TIMES; i++) {
    int error_num = test();
    std::cout << "test round=" << i << "\terror_num=" << error_num << std::endl;
    if (error_num) fail_times++;
  }
  std::cout << "fail ratio=" << (fail_times * 1.0) / TEST_TIMES << std::endl;
}