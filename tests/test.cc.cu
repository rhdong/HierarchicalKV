
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <cuda/std/semaphore>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace cooperative_groups;
namespace cg = cooperative_groups;
using namespace std;

using ValueType = int;

constexpr size_t DIM = 16;
constexpr size_t num_vectors_per_slice = 8 * 16777216;
constexpr size_t num_vector_per_bucket = 128;

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
  uint64_t* keys;      // ignore it!
  uint64_t* metas;     // ignore it!
  ValueType* cache;    // ignore it!
  ValueType* vectors;  // <<<----important member
  uint64_t cur_meta;   // ignore it!
  uint64_t min_meta;   // ignore it!
  int min_pos;         // ignore it!
};

__global__ void write_read(Bucket* buckets, int bucket_idx,
                           const ValueType val) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  ValueType* vectors = buckets[bucket_idx].vectors;
  *(vectors + tid * DIM) = val;
}

__global__ void read_when_error(Bucket* buckets, int bucket_idx,
                                int vector_idx) {
  ValueType* vectors = buckets[bucket_idx].vectors;
  ValueType val = *(vectors + vector_idx * DIM);
  printf("device view: ptr=%p\tval=%d\n", (vectors + vector_idx * DIM), val);
}

int main() {
  int num_slices = 1;
//  ValueType** slices;
  size_t slice_size = num_vectors_per_slice * sizeof(ValueType) * DIM;
//  cudaMallocManaged(&slices, sizeof(ValueType*) * num_slices);

  int num_buckets = num_vectors_per_slice / num_vector_per_bucket;
  Bucket* buckets;
  size_t bucket_size = num_vector_per_bucket * sizeof(ValueType) * DIM;
  cudaMallocManaged(&buckets, sizeof(Bucket) * num_buckets);

  std::cout << "size of Bucket=" << sizeof(Bucket) << std::endl;

  assert(num_buckets == (1024 * 1024));
  assert(slice_size == (8ul << 30));
  assert(bucket_size == (128 * 4 * 16));
  assert(slice_size == (bucket_size * num_buckets));

  ValueType* slice;
  CUDA_CHECK(cudaHostAlloc(&slice, slice_size,
                           cudaHostAllocMapped | cudaHostAllocWriteCombined));
//  slices[0] = slice;

  for (int i = 0; i < num_buckets; i++) {
    ValueType* h_slice = slice + (num_vector_per_bucket * DIM * i);
    CUDA_CHECK(cudaHostGetDevicePointer(&(buckets[i].vectors), h_slice, 0));
  }
  std::cout << "finish allocating"
            << ", num_buckets=" << num_buckets << ", slice_size=" << (8ul << 30)
            << ", bucket_size=" << (128 * 4 * 16) << std::endl;

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

  size_t error_num = 0;
  size_t correct_num = 0;
  for (int i = 0; i < num_buckets; i++) {
    for (int j = 0; j < num_vector_per_bucket; j++) {
      ValueType val = slice[i * num_vector_per_bucket * DIM + j * DIM];
      if (val != magic_numbers) {
        read_when_error<<<1, 1, 0, stream>>>(buckets, i, j);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("host   view: ptr=%p\tval=%d\n\n",
               (slice + i * num_vector_per_bucket * DIM + j * DIM), val);
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

  CUDA_CHECK(cudaFreeHost(slice));
//  CUDA_CHECK(cudaFree(slices));
  CUDA_CHECK(cudaFree(buckets));
}