// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/args/reduce_scatter.h"
#include "flux/args/all_gather.h"
#include "flux/utils.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

#ifdef FLUX_SHM_USE_NVSHMEM
#include "nvshmemx.h"
#endif

#include <cuda_runtime_api.h>
#include <cuda.h>

using namespace bytedance::flux;

#define CUStreamWriteValue(...) cuda_stub().cuStreamWriteValue32_v2(__VA_ARGS__)
#define SPLIT 1

typedef struct {
    const void * input;
    const void * weight;
    const void * bias;
    const void * input_scale;
    const void * weight_scale;
    const void * output_scale;
    void * gemm_buffer;
    void ** reduce_buffers;
    void ** output_scatter_ptrs;
    void ** barrier_ptrs;
    void ** output_buffers;
    void ** barrier_buffers;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t wk;
    int32_t nnodes;
    int32_t max_m;
    int32_t n_dim;
    int32_t rank;
    int32_t world_size;
    int32_t local_world_size;
    int32_t local_rank;
    int32_t node_idx;
    int32_t num_blocks;
    int32_t n_split;
    bool fast_accum;
    bool is_bf16;
    bool transpose_weight;
    bool fuse_reduction;
    bool use_1d_ring;
    bool use_p2p_read;
    bool is_fp8_gemm;
    bool use_barrier_queue;
    bool use_gemmk;
    bool use_cudaMemcpyAsync;
    bool per_tile_flags;
    bool no_nvlink;
    cudaStream_t stream;
    cudaStream_t rs_stream;
    cudaEvent_t event;
    bool has_bias() const {return bias != nullptr;}
    DataTypeEnum input_dtype() const {
      if (is_bf16) return _BF16{};
      else return  _FP16{};
    }
    DataTypeEnum output_dtype() const {return input_dtype();}
} GemmRSParams;

auto
get_gemm_meta(GemmRSParams params) {
  ArchEnum arch = get_arch();
  auto gemm_layout = params.transpose_weight ? _RRR{}() : _RCR{}();
  auto input_dtype = params.input_dtype();
  auto output_dtype = params.output_dtype();
  auto dt_conf = make_gemm_dtype_config(
      input_dtype, input_dtype, params.has_bias() ? output_dtype : _Void{}(), output_dtype);

  bool fast_accum = params.fast_accum & dt_conf.is_input_fp8();
  bool is_gemm_v2 = ((int)arch < (int)_Sm90{}());
  auto meta = make_gemm_meta(
      dt_conf,
      arch,
      _ReduceScatter{},
      gemm_layout,
      is_gemm_v2 ? _GemmV2{}() : _GemmV3{}(),
      is_gemm_v2 ? UnifiedImplMeta(make_gemm_v2_meta(fast_accum))
                 : UnifiedImplMeta(make_gemm_v3_meta(fast_accum)),
      make_reduce_scatter_meta(
          params.fuse_reduction,
          params.nnodes > 1        ? _AcrossNode{}()
          : params.no_nvlink ? _IntraNodePcie{}()
                            : _IntraNode{}()));
  return meta;
}

RuntimeConfig
get_rt_conf(GemmRSParams params) {
  // row major for streamk, todo: make weight layout an option
  FLUX_CHECK_LE(params.m, params.max_m) << "m-dim greater than maximum possible value";
  FLUX_CHECK_EQ(params.n, params.n_dim) << "n-dim != expected n_dim";
  FLUX_CHECK_EQ(params.wk, params.k) << "weight k-dim mismatch";
  return make_runtime_config(params.m,
                             params.n,
                             params.k,
                             make_reduce_scatter_runtime_config(
                             params.world_size,
                             params.nnodes));
}


typedef struct {
  void * input;
  void * input_buffer;
  void * weight;
  void * bias;
  void * output_buffer;
  void * barrier_buffer;
  void * gemm_buffer;
  cudaStream_t current_stream;
  cudaEvent_t ready_event;
  int32_t n;
  int32_t k;
  int32_t n_dim;
  int32_t k_dim;
  int32_t input_size_0;
  int32_t rank;
  int32_t world_size;
  int32_t nnodes;
  int32_t ring_mode;
  bool is_bf16;
  bool kDebugRunGemm;
  bool transpose_weight;
  bool fast_accum;
  bool has_bias() const {return bias != nullptr;}
  DataTypeEnum input_dtype() const {
    if (is_bf16) return _BF16{};
    else return  _FP16{};
  }
  DataTypeEnum output_dtype() const {return input_dtype();}
} AGGemmParams;

auto
get_gemm_meta(AGGemmParams params) {
  ArchEnum arch = get_arch();
  auto input_dtype = params.input_dtype();
  auto output_dtype = params.output_dtype();
  auto dtype_config = make_gemm_dtype_config(
      input_dtype, input_dtype, params.has_bias() ? output_dtype : _Void{}(), output_dtype);

  auto gemm_layout = params.transpose_weight ? _RRR{}() : _RCR{}();
  UnifiedImplMeta impl_spec = None{};

  bool use_fast_accum = params.fast_accum and dtype_config.is_input_fp8();
  auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
  if (impl == _GemmV2{}) {
    impl_spec = make_gemm_v2_meta(use_fast_accum);
  } else if (impl == _GemmV3{}) {
    impl_spec = make_gemm_v3_meta(use_fast_accum);
  }

  auto meta = make_gemm_meta(dtype_config, arch, _AGKernel{}, gemm_layout, impl, impl_spec);
  return meta;
}

RuntimeConfig
get_rt_config(AGGemmParams params) {
  FLUX_CHECK(params.n == params.n_dim) << "n-dim != expected n_dim: " << params.n << " vs " << params.n_dim;
  FLUX_CHECK(params.k == params.k_dim) << "weight k-dim mismatch: " << params.k << " != " << params.k_dim;

  return make_runtime_config(
      params.input_size_0 * params.world_size,
      params.n_dim,
      params.k_dim,
      make_all_gather_runtime_config(params.world_size, params.nnodes, (int)(params.ring_mode)));
}


#ifdef __cplusplus
extern "C" {
#endif

void set_ready(int32_t* barrier_ptr, int segment, int split_index, cudaStream_t stream) {
    CU_CHECK(CUStreamWriteValue(
        stream,
        (CUdeviceptr)(barrier_ptr + (segment * SPLIT + split_index)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
}

void ensure_nvml_init_capi() {
  ensure_nvml_init();
}

const char * get_gpu_device_name_capi(int devid) {
  return get_gpu_device_name(devid);
}

void cudaipc_barrier_all_on_stream_impl_capi(cudaStream_t stream, int32_t **sync_buffer_ptr, int rank, int world_size) {
    cudaipc_barrier_all_on_stream_impl(stream, sync_buffer_ptr, rank, world_size);
}

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
               const bool check_can_implement,
               cudaStream_t stream,
               cudaStream_t rs_stream,
               cudaEvent_t event) {
  GemmRSParams params {
      .input = input,
      .weight = weight,
      .bias = bias,
      .input_scale = input_scale,
      .weight_scale = weight_scale,
      .output_scale = output_scale,
      .gemm_buffer = gemm_buffer,
      .reduce_buffers = reduce_buffers,
      .output_scatter_ptrs = output_scatter_ptrs,
      .barrier_ptrs = barrier_ptrs,
      .output_buffers = output_buffers,
      .barrier_buffers = barrier_buffers,
      .m = m, .n = n, .k = k, .wk=wk,
      .nnodes = nnodes,
      .max_m = max_m,
      .n_dim = n_dim,
      .rank = rank,
      .world_size = world_size,
      .local_world_size = local_world_size,
      .local_rank = local_rank,
      .node_idx = node_idx,
      .num_blocks = num_blocks,
      .n_split = n_split,
      .fast_accum = fast_accum,
      .is_bf16 = is_bf16,
      .transpose_weight = transpose_weight,
      .fuse_reduction = fuse_reduction,
      .use_1d_ring = use_1d_ring,
      .use_p2p_read = use_p2p_read,
      .is_fp8_gemm = is_fp8_gemm,
      .use_barrier_queue = use_barrier_queue,
      .use_gemmk = use_gemmk,
      .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
      .per_tile_flags = per_tile_flags,
      .no_nvlink = no_nvlink,
      .stream = stream,
      .rs_stream = rs_stream,
      .event = event
  };
  auto meta = get_gemm_meta(params);
  auto rt_conf = get_rt_conf(params);
  // get cutlass op
  OpRegistry::OpPtr cutlass_op;
  cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);

  ReduceScatterArguments reduce_scatter_args{
      .reduce_scatter_num_blocks = num_blocks,
      .rs_stream = rs_stream,
      .event = event,
      .use_barrier_queue = use_barrier_queue,
      .use_gemmk = use_gemmk,
      .per_tile_flags = per_tile_flags,
      .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
      .n_split = n_split,
      .sub_world_size = 1,
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
      .opaque = nccl_comm,
#else
      .opaque = nullptr,
#endif
      .use_1d_ring = use_1d_ring,
      .use_p2p_read = use_p2p_read,
  };

  FLUX_CHECK(input_scale == nullptr);
  FLUX_CHECK(weight_scale == nullptr);
  FLUX_CHECK(output_scale == nullptr);

  const GemmReduceScatterArguments args{
      .m = rt_conf.m(),
      .n = rt_conf.n(),
      .k = rt_conf.k(),
      .rank = static_cast<int>(rank),
      .world_size = static_cast<int>(world_size),
      .nnodes = static_cast<int>(nnodes),
      .alpha = 1.0f,
      .beta = bias != nullptr ? 1.0f : 0.0f,
      .input = input,
      .weight = weight,
      .bias = bias,
      .output_scatter_ptrs = output_scatter_ptrs,
      .local_reduce_buffer = reduce_buffers[rank],
      .barrier_ptrs = barrier_ptrs,
      .avail_sms = params.no_nvlink ? 1 : -1,
      .reduce_scatter_args = reduce_scatter_args
  };

  if(check_can_implement) {
    if(cutlass_op->can_implement(args)) return 1;
    else return 0;
  }

  // initialize workspace
  int64_t workspace_size = cutlass_op->get_workspace_size(args);
  if (get_workspace_size_flag)
      return workspace_size;
  void *workspace = gemm_buffer;

  // initialize barrier workspace
  int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
  // * 8 is for corner case reduce_scatter tiles. never mind this won't be a large memory
  barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags) * 8;
  if (get_barrier_workspace_size)
      return barrier_workspace_size;

  cutlass_op->run(args, workspace, stream);

  return 0;
}

size_t ag_gemm(void * input,
               void * input_buffer,
               void * weight,
               void * bias,
               void * output_buffer,
               void * barrier_buffer,
               void * gemm_buffer,
               cudaStream_t current_stream,
               cudaEvent_t ready_event,
               const int32_t n,
               const int32_t k,
               const int32_t n_dim,
               const int32_t k_dim,
               const int32_t input_size_0,
               const int32_t rank,
               const int32_t world_size,
               const int32_t nnodes,
               const int32_t ring_mode,
               const bool is_bf16,
               const bool kDebugRunGemm,
               const bool transpose_weight,
               const bool fast_accum,
               const bool return_workspace_size,
               const bool check_can_implement) {
  AGGemmParams params;

  params.input = input;
  params.input_buffer = input_buffer;
  params.weight = weight;
  params.bias = bias;
  params.output_buffer = output_buffer;
  params.barrier_buffer = barrier_buffer;
  params.gemm_buffer = gemm_buffer;
  params.current_stream = current_stream;
  params.ready_event = ready_event;
  params.n = n;
  params.k = k;
  params.n_dim = n_dim;
  params.k_dim = k_dim;
  params.input_size_0 = input_size_0;
  params.rank = rank;
  params.world_size = world_size;
  params.nnodes = nnodes;
  params.ring_mode = ring_mode;
  params.is_bf16 = is_bf16;
  params.kDebugRunGemm = kDebugRunGemm;
  params.transpose_weight = transpose_weight;
  params.fast_accum = fast_accum;

  auto meta = get_gemm_meta(params);
  auto rt_config = get_rt_config(params);
  auto hparams = OpRegistry::instance().get_hparams(meta, rt_config);
  auto cutlass_op = OpRegistry::instance().get_op(meta, hparams);
  auto gemm_args = AGKernelArguments{
        .m = rt_config.m(),
        .n = rt_config.n(),
        .k = rt_config.k(),
        .rank = static_cast<int>(params.rank),
        .world_size = static_cast<int>(params.world_size),
        .nnodes = static_cast<int>(params.nnodes),
        .alpha = 1.0f,
        .beta = params.has_bias() ? 1.0f : 0.0f,
        .input = params.input,
        .input_buffer = params.input_buffer,
        .weight = params.weight,
        .bias = params.bias,
        .output = params.output_buffer,
        .barrier_buffer = params.barrier_buffer};

  if(check_can_implement) {
    if(cutlass_op->can_implement(gemm_args)) return 1;
    else return 0;
  }
  // AG Gemm Workspace
  int64_t workspace_size = cutlass_op->get_workspace_size(gemm_args);
  if (return_workspace_size) return workspace_size;

  /// GEMM
  if (params.kDebugRunGemm) {
    cutlass_op->run(gemm_args, params.gemm_buffer, params.current_stream);
  } else {
    CUDA_CHECK(cudaStreamWaitEvent(params.current_stream, params.ready_event));
  }
  return 0;
}
#ifdef __cplusplus
}
#endif
