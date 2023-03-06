#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

using ValueType = int;
constexpr int DIM = 16;

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
  uint64_t* a;      // ignore it!
  uint64_t* b;     // ignore it!
  ValueType* c;    // ignore it!
  ValueType* vectors;  // <<<----important member
  uint64_t d;   // ignore it!
  uint64_t e;   // ignore it!
  int f;         // ignore it!
};

__global__ void write_read(Bucket* buckets, int bucket_idx,
                           const ValueType val) {
  int vector_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  ValueType* vectors = buckets[bucket_idx].vectors;
  *(vectors + tid * DIM) = bucket_idx * num_vector_per_bucket + vector_idx;
}

__global__ void read_when_error(Bucket* buckets, int bucket_idx,
                                int vector_idx) {
  ValueType* vectors = buckets[bucket_idx].vectors;
  ValueType val = *(vectors + vector_idx * DIM);
  printf("device view: ptr=%p\tval=%d\n", (vectors + vector_idx * DIM), val);
}


constexpr size_t num_vector_per_bucket = 128;
constexpr size_t num_buckets = 1024 * 1024;
constexpr size_t num_vector = num_buckets * num_vector_per_bucket;

constexpr size_t memory_pool_size = num_vector * sizeof(ValueType) * DIM; // = 128 * 1024 * 1024 * 4 * 16 = 8GB
constexpr size_t bucket_size = num_vector_per_bucket * sizeof(ValueType) * DIM;


int main() {
  Bucket* buckets;

  // Allocating the buckets structs.
  cudaMallocManaged(&buckets, sizeof(Bucket) * num_buckets);

  std::cout << "size of Bucket=" << sizeof(Bucket) << std::endl;

  assert(num_buckets == (1024 * 1024));
  assert(memory_pool_size == (8ul << 30));
  assert(bucket_size == (128 * 4 * 16));
  assert(memory_pool_size == (bucket_size * num_buckets));

  // Allocating a memory pool on host memory for all of the vectors.
  ValueType* host_memory_pool;
  CUDA_CHECK(cudaHostAlloc(&host_memory_pool, memory_pool_size,
                           cudaHostAllocMapped | cudaHostAllocWriteCombined));

  // Make the `vectors` pointer of each bucket point to different section of the memory pool.
  for (int i = 0; i < num_buckets; i++) {
    ValueType* h_memory_pool = host_memory_pool + (num_vector_per_bucket * DIM * i);
    CUDA_CHECK(cudaHostGetDevicePointer(&(buckets[i].vectors), h_memory_pool, 0));
  }
  std::cout << "finish allocating"
            << ", num_buckets=" << num_buckets << ", memory_pool_size=" << (8ul << 30)
            << ", bucket_size=" << (128 * 4 * 16) << std::endl;

  // Write magic_numbers to first element of each `vectors`.
  // BTW, each `vectors` points to a section of memory pool with 8192 bytes, and
  // each element of the `vectors` has 16 floats (64 bytes).
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  ValueType magic_numbers = 88;
  for (int i = 0; i < num_buckets; i++) {
    write_read<<<1, num_vector_per_bucket, 0, stream>>>(buckets, i,
                                                        magic_numbers);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "finish writing" << std::endl;

  // Checking if the value is expected from both of device and host sides.
  size_t error_num = 0;
  size_t correct_num = 0;
  for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
    for (int vector_idx = 0; vector_idx < num_vector_per_bucket; vector_idx++) {
      ValueType val = host_memory_pool[bucket_idx * num_vector_per_bucket * DIM + vector_idx * DIM];
      if (val != (bucket_idx * num_vector_per_bucket + vector_idx)) {
        read_when_error<<<1, 1, 0, stream>>>(buckets, bucket_idx, vector_idx);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("host   view: ptr=%p\tval=%d\n\n",
               (host_memory_pool + bucket_idx * num_vector_per_bucket * DIM + vector_idx * DIM), val);
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
  std::cout << "finish checking" << std::endl;

  CUDA_CHECK(cudaFreeHost(host_memory_pool));
  CUDA_CHECK(cudaFree(buckets));
}