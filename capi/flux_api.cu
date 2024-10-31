#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/helper_kernels.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/args/reduce_scatter.h"
#include "flux/utils.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

#ifdef FLUX_SHM_USE_NVSHMEM
#include "nvshmemx.h"
#endif

#include <cuda_runtime_api.h>

using namespace bytedance::flux;

#if 0
using bytedance::flux::DataTypeEnum;
using bytedance::flux::_BF16;
using bytedance::flux::_FP16
using bytedance::flux::ArchEnum;
using bytedance::flux::_RRR;
using bytedance::flux::_RCR;
using bytedance::flux::_Void
using bytedance::flux::_Sm90;
#endif

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
#if 0
  // TODO(umiswing): add check in phi
  CHECK_INPUT(input, params.input_dtype);
  CHECK_INPUT(weight, params.input_dtype);
  TORCH_CHECK(input.dim() == 2, "input dim is not 2");
  TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
#endif
#if 0
  if (bias.has_value()) {
    CHECK_INPUT(bias.value(), params.output_dtype);
    TORCH_CHECK(bias->dim() == 2, "bias dim is not 2");
    TORCH_CHECK(
        m == bias->size(0),
        "bias dim0 != m: " + std::to_string(bias->size(0)) + " vs " + std::to_string(m));
    TORCH_CHECK(
        n == bias->size(1),
        "bias dim1 != n: " + std::to_string(bias->size(1)) + " vs " + std::to_string(n));
  }
#endif

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

size_t gemm_rs_internal(const void * const input,
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
       // umiswing: no other place use sub_world_size, just keep it as place holder
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
  // cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // TODO(umiswing): add fp8 kernel
  // if (!is_fp8_gemm)
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

    // printf("\nupdate so success\n");
    cutlass_op->run(args, workspace, stream);

    // if (get_barrier_workspace_size || get_workspace_size_flag) return 1;
#if 0
    if (get_arch() == _Sm90{} and nnodes == 1) {
      // only local reduce, skip nvshmem barrier
    } else {
    // printf("\nbegin barrier all on stram after comment out\n");
      // TODO(umiswing): add this for ampere
      // flux_barrier_all_on_stream(stream, sync_buffers, rank);
      cudaipc_barrier_all_on_stream_impl(stream, sync_buffer_ptrs, rank, world_size);
    }
#endif
    return 0;
}
#ifdef __cplusplus
extern "C" {
#endif

void ensure_nvml_init_capi() {
  ensure_nvml_init();
  printf("\n>>>>>>>>>>>>>>>>>>>>>>>>arrvie at init\n");
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
               cudaStream_t stream,
               cudaStream_t rs_stream,
               cudaEvent_t event) {
  return gemm_rs_internal(
             input,
             weight,
             bias,
             input_scale,
             weight_scale,
             output_scale,
             gemm_buffer,
             reduce_buffers,
             output_scatter_ptrs,
             barrier_ptrs,
             output_buffers,
             barrier_buffers,
             sync_buffer_ptrs,
             m,
             n,
             k,
             wk,
             nnodes,
             max_m,
             n_dim,
             rank,
             world_size,
             local_world_size,
             local_rank,
             node_idx,
             num_blocks,
             n_split,
             fast_accum,
             is_bf16,
             transpose_weight,
             fuse_reduction,
             use_1d_ring,
             use_p2p_read,
             is_fp8_gemm,
             use_barrier_queue,
             use_gemmk,
             use_cudaMemcpyAsync,
             per_tile_flags,
             no_nvlink,
             get_workspace_size_flag,
             get_barrier_workspace_size,
             stream,
             rs_stream,
             event);
}
#ifdef __cplusplus
}
#endif

#if 0
// TODO(umiswing): place holder, no need on single node
void forward_reduce_scatter_impl() {
  return;
}
#endif
