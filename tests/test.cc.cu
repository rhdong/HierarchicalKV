
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
constexpr size_t TILE_SIZE = 4;

template <class V, uint32_t TILE_SIZE = 4>
__device__ __forceinline__ void copy_vector(
    cg::thread_block_tile<TILE_SIZE> const& g, const V val, V* dst,
    const size_t dim) {
  for (auto i = g.thread_rank(); i < dim; i += g.size()) {
    dst[i] = val;
  }
}

template <class K, class V, class M, uint32_t TILE_SIZE = 4>
__global__ void upsert_kernel_with_io_core(V** slices, int* index, size_t N) {

  size_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto g = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  int rank = g.thread_rank();


  for (size_t t = tid; t < N; t += blockDim.x * gridDim.x) {
    size_t vector_idx = index[t / TILE_SIZE];
    const V val= static_cast<V>(vector_idx * 0.00001);
    size_t target_slice = vector_idx / num_vectors_per_slice;
    size_t target_offset = (vector_idx % num_vectors_per_slice) * DIM;
    copy_vector<V, TILE_SIZE>(g, val, slices[target_slice] + target_offset, DIM);
  }

};

static inline size_t SAFE_GET_GRID_SIZE(size_t N, int block_size) {
  return  (((N)-1) / block_size + 1);
};

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
  AtomicKey<K>* keys;    // HBM
  AtomicMeta<M>* metas;  // HBM
  V* cache;              // HBM(optional)
  V* vectors;            // Pinned memory or HBM

  /* For upsert_kernel without user specified metas
     recording the current meta, the cur_meta will
     increment by 1 when a new inserting happens. */
  AtomicMeta<M> cur_meta;

  /* min_meta and min_pos is for or upsert_kernel
     with user specified meta. They record the minimum
     meta and its pos in the bucket. */
  AtomicMeta<M> min_meta;
  AtomicPos<int> min_pos;
};

template <class K, class V, class M>
__global__ void write_read(Bucket<K, V, M>* buckets, int bucket_idx, int vector_idx, const V val) {
  V* vectors = buckets[bucket_idx].vectors;
  *(vectors + vector_idx * DIM) = val;
}

int main() {
  int num_slices = 1;
  V** slices;
  size_t slice_size = num_vectors_per_slice * sizeof(V) * DIM;
  cudaMallocManaged(&slices, sizeof(V*) * num_slices);

  int num_buckets = num_vectors_per_slice / num_vector_per_bucket;
  Bucket<K, V, M>* buckets;
  size_t bucket_size = num_vector_per_bucket * sizeof(V) * DIM;
  cudaMallocManaged(&buckets, sizeof(Bucket<K, V, M>*) * num_buckets);

  assert(num_buckets == (1024 * 1024));
  assert(slice_size == (8ul << 30));
  assert(bucket_size == (128 * 4 * 16));

//  int* d_index;
  V* slice;
//  cudaMalloc(&d_index, num_vectors_per_slice * num_slices * sizeof(int));
  cudaMallocHost(&slice, slice_size, cudaHostAllocMapped | cudaHostAllocWriteCombined);
  slices[0] = slice;

  for(int i = 0; i < num_buckets; i++){
    buckets[i].vectors = slices[0] + (num_vector_per_bucket * DIM * i);
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  int magic_numbers = 88;
  for(int i = 0; i < num_buckets; i++){
    for(int j = 0; j < num_vector_per_bucket; j++){
      write_read<K, V, M><<<1, 1, 0, stream>>>(buckets, i, j, magic_numbers);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }

  for(int i = 0; i < num_buckets; i++){
    for(int j = 0; j < num_vector_per_bucket; j++){
      assert(buckets[i].vectors[j * DIM] == magic_numbers);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

//  for(int i = 0; i <  num_vectors_per_slice * num_slices; i++){
//    h_index[i] = i;
//  }
//  cudaMemcpy(d_index, h_index, num_vectors_per_slice * num_slices * sizeof(int), cudaMemcpyHostToDevice);

//  thrust::default_random_engine g;
//  thrust::shuffle(thrust::device, d_index, d_index + num_vectors_per_slice * num_slices, g);


//  const size_t block_size = 128;
//  const size_t N = num_slices * num_vectors_per_slice * TILE_SIZE;
//  const size_t grid_size = SAFE_GET_GRID_SIZE(N, block_size);
//
//  upsert_kernel_with_io_core<K, V, M, 4><<<grid_size, block_size, 0, 0>>>(slices, d_index, N);
//  CUDA_CHECK(cudaDeviceSynchronize());
//
//  for(int i = 0; i < num_slices; i++){
//    V* slice = slices[i];
//    for(int j = 0; j < num_vectors_per_slice; j++){
//      float expected = static_cast<V>((i * num_vectors_per_slice + j) * 0.00001);
//      if(expected != slice[j * DIM]){
//        std::cout << expected << " " << slice[j * DIM] << std::endl;
//      }
//    }
//  }

  for(int i = 0; i < num_slices; i++){
    cudaFreeHost(slices[i]);
  }
  cudaFree(slices);
  cudaFree(buckets);
//  cudaFree(d_index);
//  cudaFreeHost(h_index);
}