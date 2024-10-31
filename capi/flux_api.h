#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t gemm_rs(const void * const input,
               const void * const weight,
               const void * const bias,
               const void * const input_scale,
               const void * const weight_scale,
               const void * const output_scale,
               void * const gemm_buffer,
               void ** reduce_buffers,
               void ** output_scatter_ptrs,
               void ** barrier_ptrs,
               void ** output_buffers,
               void ** barrier_buffers,
               int32_t ** sync_buffer_ptrs,
               const int32_t m,
               const int32_t n,
               const int32_t k,
               const int32_t wk,
               const int32_t nnodes,
               const int32_t max_m,
               const int32_t n_dim,
               const int32_t rank,
               const int32_t world_size,
               const int32_t local_world_size,
               const int32_t local_rank,
               const int32_t node_idx,
               const int32_t num_blocks,
               const int32_t n_split,
               const bool fast_accum,
               const bool is_bf16,
               const bool transpose_weight,
               const bool fuse_reduction,
               const bool use_1d_ring,
               const bool use_p2p_read,
               const bool is_fp8_gemm,
               const bool use_barrier_queue,
               const bool use_gemmk,
               const bool use_cudaMemcpyAsync,
               const bool per_tile_flags,
               const bool no_nvlink,
               const bool get_workspace_size_flag,
               const bool get_barrier_workspace_size,
               cudaStream_t stream,
               cudaStream_t rs_stream,
               cudaEvent_t event);

void ensure_nvml_init_capi();

const char * get_gpu_device_name_capi(int devid);

void cudaipc_barrier_all_on_stream_impl_capi(cudaStream_t stream, int32_t **sync_buffer_ptr, int rank, int world_size);
#ifdef __cplusplus
}
#endif
