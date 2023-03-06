
#include <iostream>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <cuda/std/semaphore>

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace cooperative_groups;
namespace cg = cooperative_groups;
using namespace std;


using K = uint64_t;
using M = uint64_t;
using V = int;


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

#define CUDA_CHECK(val)                                 \
  do {                                                  \
    cuda_check_((val), __FILE__, __LINE__); \
  } while (0)


template <class K>
using AtomicKey = cuda::atomic<K, cuda::thread_scope_device>;

template <class M>
using AtomicMeta = cuda::atomic<M, cuda::thread_scope_device>;

template <class T>
using AtomicPos = cuda::atomic<T, cuda::thread_scope_device>;


template <class K, class V, class M>
struct Bucket {
  AtomicKey<K>* keys;    // ignore it!
  AtomicMeta<M>* metas;  // ignore it!
  V* cache;              // ignore it!
  V* vectors;            // <<<----important
  AtomicMeta<M> cur_meta; // ignore it!
  AtomicMeta<M> min_meta; // ignore it!
  AtomicPos<int> min_pos; // ignore it!
};

template <class K, class V, class M>
__global__ void write_read(Bucket<K, V, M>* buckets, int bucket_idx, const V val) {
  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  V* vectors = buckets[bucket_idx].vectors;
  *(vectors + tid * DIM) = val;
}

template <class K, class V, class M>
__global__ void read_when_error(Bucket<K, V, M>* buckets, int bucket_idx, int vector_idx) {
  V* vectors = buckets[bucket_idx].vectors;
  V val = *(vectors + vector_idx * DIM);
  printf("device view: ptr=%p\tval=%d\n", (vectors + vector_idx * DIM), val);
}

int main() {
  int num_slices = 1;
  V** slices;
  size_t slice_size = num_vectors_per_slice * sizeof(V) * DIM;
  cudaMallocManaged(&slices, sizeof(V*) * num_slices);

  int num_buckets = num_vectors_per_slice / num_vector_per_bucket;
  Bucket<K, V, M>* buckets;
  size_t bucket_size = num_vector_per_bucket * sizeof(V) * DIM;
  cudaMallocManaged(&buckets, sizeof(Bucket<K, V, M>) * num_buckets);

  assert(num_buckets == (1024 * 1024));
  assert(slice_size == (8ul << 30));
  assert(bucket_size == (128 * 4 * 16));
  assert(slice_size == (bucket_size * num_buckets));

  V* slice;
  CUDA_CHECK(cudaHostAlloc(&slice, slice_size, cudaHostAllocMapped | cudaHostAllocWriteCombined));
  slices[0] = slice;

  for(int i = 0; i < num_buckets; i++){
    V* h_slice = slices[0] + (num_vector_per_bucket * DIM * i);
    CUDA_CHECK(cudaHostGetDevicePointer(&(buckets[i].vectors), h_slice, 0));
  }
  std::cout << "finish allocating" << ", num_buckets=" << num_buckets
            << ", slice_size=" << (8ul << 30)
            << ", bucket_size=" << (128 * 4 * 16) << std::endl;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  V magic_numbers = 88;
  for(int i = 0; i < num_buckets; i++){
    write_read<K, V, M><<<1, num_vector_per_bucket, 0, stream>>>(buckets, i, magic_numbers);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "finish writing" << std::endl;

  size_t error_num = 0;
  size_t correct_num = 0;
  for(int i = 0; i < num_buckets; i++){
    for(int j = 0; j < num_vector_per_bucket; j++){
      V val = slice[i * num_vector_per_bucket * DIM + j * DIM];
      if(val != magic_numbers) {
        read_when_error<K, V, M><<<1, 1, 0, stream>>>(buckets, i, j);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("host   view: ptr=%p\tval=%d\n\n", (slice + i * num_vector_per_bucket * DIM + j * DIM), val);
        error_num++;
      } else {
        correct_num++;
      }
    }
  }
  std::cout << "error_num=" << error_num << "\tcorrect_num=" << correct_num << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::cout << "finish checking" << std::endl;


  CUDA_CHECK(cudaFreeHost(slice));
  CUDA_CHECK(cudaFree(slices));
  CUDA_CHECK(cudaFree(buckets));
}