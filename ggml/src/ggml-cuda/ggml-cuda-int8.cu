/*
All the code in this file is my custom implementation for testing the ggml library. 
It is not part of the original ggml library.
*/

#include "ggml.h"
#include "ggml-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <cuda_fp16.h>

#include "ggml-cuda-int8.cuh"

// constants for Q8_0 format
#define QK8_0 32

bool use_custom_kernel() {
    const char * env = std::getenv("GGML_USE_CUSTOM_KERNEL");
    return env != nullptr && std::strcmp(env, "1") == 0;
}


void print_ggml_tensor_info(const struct ggml_tensor * t, const char * name) {
    if (t == NULL) {
        printf("%s: NULL tensor\n", name);
        return;
    }

    printf("Tensor %s\n", name);
    printf("  type: %s\n", ggml_type_name(t->type));

    printf("  ne: [%lld, %lld, %lld, %lld]\n",
           (long long)t->ne[0],
           (long long)t->ne[1],
           (long long)t->ne[2],
           (long long)t->ne[3]);

    printf("  nb: [%zu, %zu, %zu, %zu]\n",
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    printf("\n");
}

__global__ void quantize_fp32_to_int8_row_wise_kernel(
    const float *input,
    int8_t *output,
    float *scales,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    extern __shared__ float sdata[];

    const float *row_input = input + row * cols;
    int8_t *row_output = output + row * cols;

    // Step 1: find max absolute value in this row
    float local_max = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float v = fabsf(row_input[col]);
        local_max = fmaxf(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_abs = sdata[0];

    // Avoid division by zero
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;

    if (tid == 0) {
        scales[row] = scale;
    }

    __syncthreads();

    // Step 2: quantize each element
    for (int col = tid; col < cols; col += blockDim.x) {
        float q = nearbyintf(row_input[col] / scale);

        q = fminf(fmaxf(q, -128.0f), 127.0f);

        row_output[col] = static_cast<int8_t>(q);
    }
}

void quantize_fp32_to_int8_row_wise_cuda(
    const float * input,
    int8_t * output,
    float * scales,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int threads = 512;
    const int blocks = rows;
    const size_t shared_mem = threads * sizeof(float);

    quantize_fp32_to_int8_row_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        input,
        output,
        scales,
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_fp32_to_int8_row_wise_cuda: %s\n", cudaGetErrorString(err));
    }
}

__global__ void quantize_fp32_to_int8_col_wise_kernel(
    const float * input,
    int8_t * output,
    float * scales,
    int rows,
    int cols
) {
    int col = blockIdx.x;
    int tid = threadIdx.x;

    if (col >= cols) {
        return;
    }

    extern __shared__ float sdata[];

    // Step 1: find max absolute value in this column
    float local_max = 0.0f;

    for (int row = tid; row < rows; row += blockDim.x) {
        float v = fabsf(input[row + col * rows]);
        local_max = fmaxf(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_abs = sdata[0];
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;

    if (tid == 0) {
        scales[col] = scale;
    }

    __syncthreads();

    // Step 2: quantize each element in this column
    for (int row = tid; row < rows; row += blockDim.x) {
        float q = nearbyintf(input[row + col * rows] / scale);

        q = fminf(fmaxf(q, -127.0f), 127.0f);

        output[row + col * rows] = static_cast<int8_t>(q);
    }
}

void quantize_fp32_to_int8_col_wise_cuda(
    const float * input,
    int8_t * output,
    float * scales,
    int rows,
    int cols,
    cudaStream_t stream
) {
    const int threads = 512;
    const int blocks = cols;
    const size_t shared_mem = threads * sizeof(float);

    quantize_fp32_to_int8_col_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        input,
        output,
        scales,
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in quantize_fp32_to_int8_col_wise_cuda: %s\n", cudaGetErrorString(err));
    }
}


__global__ void dequantize_i32_to_f32_kernel(
    const int32_t * input_i32,
    float * output_f32,
    const float * row_scales,
    const float * col_scales,
    int rows,
    int cols,
    int ldc
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) {
        return;
    }

    int idx = row + col * ldc;

    float scale = row_scales[row] * col_scales[col];

    output_f32[idx] = (float) input_i32[idx] * scale;
}

void dequantize_i32_to_f32_cuda(
    const int32_t * input_i32,
    float * output_f32,
    const float * row_scales,
    const float * col_scales,
    int rows,
    int cols,
    int ldc,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    dequantize_i32_to_f32_kernel<<<grid, block, 0, stream>>>(
        input_i32,
        output_f32,
        row_scales,
        col_scales,
        rows,
        cols,
        ldc
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in dequantize_i32_to_f32_cuda: %s\n", cudaGetErrorString(err));
    }
}


// If ggml_half is uint16_t, we reinterpret it as CUDA half.
struct block_q8_0_simple {
    uint16_t d;
    int8_t qs[QK8_0];
};

__device__ __forceinline__ float q8_0_scale_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

__global__ void convert_q8_0_to_int8_row_wise_kernel(
    const block_q8_0_simple * __restrict__ input_q8_0,
    int8_t * __restrict__ output_i8,
    float * __restrict__ output_scales,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    extern __shared__ float sdata[];

    // Q8_0 requires cols to be multiple of QK8_0.
    int blocks_per_row = cols / QK8_0;

    const block_q8_0_simple * row_blocks =
        input_q8_0 + row * blocks_per_row;

    int8_t * row_output =
        output_i8 + row * cols;

    // ------------------------------------------------------------
    // Step 1: find max abs real value in this row
    //
    // real_value = block_scale * q_value
    // ------------------------------------------------------------
    float local_max = 0.0f;

    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const block_q8_0_simple & blk = row_blocks[b];

        float d = q8_0_scale_to_float(blk.d);

        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float v = fabsf(d * (float) blk.qs[i]);
            local_max = fmaxf(local_max, v);
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_abs = sdata[0];

    // New row-wise scale for plain int8 matrix
    float row_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;

    if (tid == 0) {
        output_scales[row] = row_scale;
    }

    __syncthreads();

    // ------------------------------------------------------------
    // Step 2: convert Q8_0 blocks to plain INT8 row-wise format
    //
    // real_value = old_block_scale * old_q
    // new_q      = round(real_value / row_scale)
    // ------------------------------------------------------------
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const block_q8_0_simple & blk = row_blocks[b];

        float d = q8_0_scale_to_float(blk.d);

        int base_col = b * QK8_0;

        #pragma unroll
        for (int i = 0; i < QK8_0; ++i) {
            float real_value = d * (float) blk.qs[i];

            float q = nearbyintf(real_value / row_scale);
            q = fminf(fmaxf(q, -128.0f), 127.0f);

            row_output[base_col + i] = static_cast<int8_t>(q);
        }
    }
}

void convert_q8_0_to_int8_row_wise_cuda(
    const void * input_q8_0,
    int8_t * output_i8,
    float * output_scales,
    int rows,
    int cols,
    cudaStream_t stream
) {
    if (cols % QK8_0 != 0) {
        fprintf(stderr, "convert_q8_0_to_int8_row_wise_cuda: cols must be multiple of QK8_0\n");
        return;
    }

    const int threads = 256;
    const int blocks = rows;
    const size_t shared_mem = threads * sizeof(float);

    convert_q8_0_to_int8_row_wise_kernel<<<blocks, threads, shared_mem, stream>>>(
        reinterpret_cast<const block_q8_0_simple *>(input_q8_0),
        output_i8,
        output_scales,
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "CUDA error in convert_q8_0_to_int8_row_wise_cuda: %s\n",
                cudaGetErrorString(err));
    }
}



#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t status = call;                                          \
    if (status != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error: %s:%d: %s\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(status));          \
      return false;                                                     \
    }                                                                   \
  } while (0)
#endif

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                                           \
  do {                                                                  \
    cutlass::Status s = status;                                         \
    if (s != cutlass::Status::kSuccess) {                               \
      fprintf(stderr, "CUTLASS error: %s:%d: status = %d\n",            \
              __FILE__, __LINE__, static_cast<int>(s));                 \
      return false;                                                     \
    }                                                                   \
  } while (0)
#endif


/*
    CUTLASS INT8 GEMM:

        input  : int8  [M, K_gemm], row-major
        weight : int8  [N, K_gemm], row-major physical memory
        output : int32 [M, N], column-major physical memory

    Logical math:

        output = input @ weight^T

        input      = [M, K]
        weight     = [N, K]
        weight^T   = [K, N]
        output     = [M, N]

    CUTLASS layout trick:

        A is RowMajor [M, K].

        B is declared as ColumnMajor [K, N].

        But the physical memory for B is row-major [N, K].
        Row-major [N, K] has the same memory layout as
        column-major [K, N].

    Output layout:

        LayoutOutput = ColumnMajor

        Therefore:

            output[m, n] is stored at output[m + n * ldc]

        This matches your previous cuBLAS output layout.
*/
template <typename TileShape, typename WarpShape, int kStages>
bool int8_matmul_cutlass_i32(
    const int8_t* input,        // [M, K_gemm], row-major
    const int8_t* weight,       // [N_gemm, K_gemm], row-major physical memory
    int32_t* output,            // [M, N_gemm], column-major physical memory
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,                    // usually ldc >= M, often ldc == M or row_diff
    cudaStream_t stream
) {
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::ColumnMajor;


  static int const kElementsPerAccess = 1;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      ElementComputeEpilogue>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      TileShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 32>,  // INT8 Tensor Core MMA
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      kStages>;

  cutlass::gemm::GemmCoord problem_size(
      M,       // GEMM M
      N_gemm,  // GEMM N
      K_gemm   // GEMM K
  );

  cutlass::MatrixCoord input_size(M, K_gemm);

  /*
      Important:

      weight is logically [N_gemm, K_gemm] row-major.

      CUTLASS sees it as ColumnMajor [K_gemm, N_gemm].

      Therefore the MatrixCoord for B is [K_gemm, N_gemm].
  */
  cutlass::MatrixCoord weight_size(K_gemm, N_gemm);

  /*
      Output is ColumnMajor [M, N_gemm] with leading dimension ldc.
  */
  cutlass::MatrixCoord output_size(M, N_gemm);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      const_cast<ElementInputA*>(input),
      LayoutInputA::packed(input_size));

  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      const_cast<ElementInputB*>(weight),
      LayoutInputB::packed(weight_size));

  cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
      output,
      LayoutOutput(ldc));

  int32_t alpha = 1;
  int32_t beta  = 0;

  typename Gemm::Arguments arguments{
      problem_size,
      input_ref,
      weight_ref,
      output_ref,
      output_ref,
      {alpha, beta},
      1
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  cutlass::Status status;

  status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    if (workspace) cudaFree(workspace);
    CUTLASS_CHECK(status);
  }

  status = gemm_op.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    if (workspace) cudaFree(workspace);
    CUTLASS_CHECK(status);
  }

  status = gemm_op(stream);

  if (workspace) {
    cudaFree(workspace);
  }

  CUTLASS_CHECK(status);

  return true;
}

bool int8_matmul_cutlass_cuda(const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,
    cudaStream_t stream
) {
    // You can tune these tile and warp shapes for better performance.
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    const int kStages = 3;

    return int8_matmul_cutlass_i32<TileShape, WarpShape, kStages>(
        input,
        weight,
        output,
        M,
        N_gemm,
        K_gemm,
        ldc,
        stream
    );
}


#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

/*
    Fused INT8 GEMM + row/column scaling + FP32 output.

    Input:
        A        : int8  [M, K_gemm], row-major
        B        : int8  [N_gemm, K_gemm], row-major physical memory
        alphaRow : float [M]
        alphaCol : float [N_gemm]

    Output:
        D        : float [M, N_gemm], column-major physical memory

    Logical math:
        D = A @ B^T

        A      = [M, K_gemm]
        B      = [N_gemm, K_gemm]
        B^T    = [K_gemm, N_gemm]
        D      = [M, N_gemm]

    Epilogue:
        D[m, n] = float(acc_i32[m, n]) * alphaRow[m] * alphaCol[n]

    Physical output layout:
        D[m, n] is stored at:

            D[m + n * ldc]

    This matches your working INT8 -> INT32 CUTLASS path and your old
    dequantization kernel.
*/
template <typename TileShape, typename WarpShape, int kStages>
bool matmul_w8a8_cutlass_f32_ptr(
    const int8_t* A,        // [M, K_gemm], row-major
    const int8_t* B,        // [N_gemm, K_gemm], row-major physical memory
    const float* alphaRow,  // [M]
    const float* alphaCol,  // [N_gemm]
    float* D,               // [M, N_gemm], column-major physical memory
    int32_t M,
    int32_t N_gemm,
    int32_t K_gemm,
    int32_t ldc,
    cudaStream_t stream
) {
    if (!A || !B || !alphaRow || !alphaCol || !D) {
        fprintf(stderr, "matmul_w8a8_cutlass_f32_ptr: null pointer input\n");
        return false;
    }

    if (M <= 0 || N_gemm <= 0 || K_gemm <= 0) {
        fprintf(stderr, "matmul_w8a8_cutlass_f32_ptr: invalid shape\n");
        return false;
    }

    if (ldc < M) {
        fprintf(stderr,
                "matmul_w8a8_cutlass_f32_ptr: invalid ldc. ldc=%d, M=%d\n",
                ldc, M);
        return false;
    }

    if (K_gemm % 32 != 0) {
        fprintf(stderr,
                "matmul_w8a8_cutlass_f32_ptr: K_gemm must be multiple of 32. K_gemm=%d\n",
                K_gemm);
        return false;
    }

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementScale = float;
    using ElementC = float;
    using ElementOutput = float;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16 int8
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16 int8

    // Safe scalar FP32 output store
    constexpr int AlignmentC = 1;

    constexpr int EVTEpilogueStages = 1;

    using namespace cute;

    using OutputTileThreadMap =
        cutlass::epilogue::threadblock::OutputTileThreadLayout<
            TileShape,
            WarpShape,
            ElementC,
            AlignmentC,
            EVTEpilogueStages>;

    using Accum =
        cutlass::epilogue::threadblock::VisitorAccFetch;

    /*
        alphaRow[m]

        This broadcasts one scale per output row.
    */
    using RowScaleBroadcast =
        cutlass::epilogue::threadblock::VisitorColBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_1, _0, int32_t>>;

    /*
        alphaCol[n]

        This broadcasts one scale per output column.
    */
    using ColScaleBroadcast =
        cutlass::epilogue::threadblock::VisitorRowBroadcast<
            OutputTileThreadMap,
            ElementScale,
            cute::Stride<_0, _1, int32_t>>;

    /*
        First multiply:

            acc * alphaRow[m]
    */
    using ComputeRowScale =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTRowScale =
        cutlass::epilogue::threadblock::Sm80EVT<
            ComputeRowScale,
            Accum,
            RowScaleBroadcast>;

    /*
        Second multiply:

            (acc * alphaRow[m]) * alphaCol[n]
    */
    using ComputeColScale =
        cutlass::epilogue::threadblock::VisitorCompute<
            cutlass::multiplies,
            ElementCompute,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTRowColScale =
        cutlass::epilogue::threadblock::Sm80EVT<
            ComputeColScale,
            EVTRowScale,
            ColScaleBroadcast>;

    /*
        Store FP32 output in column-major physical layout:

            D[m, n] -> D[m + n * ldc]

        Therefore stride is:

            stride_m     = 1
            stride_n     = ldc
            stride_batch = ldc * N_gemm
    */
    using StoreD =
        cutlass::epilogue::threadblock::VisitorAuxStore<
            OutputTileThreadMap,
            ElementOutput,
            cutlass::FloatRoundStyle::round_to_nearest,
            cute::Stride<_1, int64_t, int64_t>>;

    using EVTD =
        cutlass::epilogue::threadblock::Sm80EVT<
            StoreD,
            EVTRowColScale>;

    using Kernel =
        typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
            ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
            ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
            ElementC, LayoutC, AlignmentC,
            ElementAccumulator,
            ElementCompute,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            TileShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 32>,
            EVTD,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            kStages,
            cutlass::arch::OpMultiplyAddSaturate,
            EVTEpilogueStages
        >::GemmKernel;

    using DeviceGemm =
        cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename EVTD::Arguments callback_args{
        {
            {
                {},
                {alphaRow, ElementScale(0), {_1{}, _0{}, int32_t(M)}},
                {}
            },
            {alphaCol, ElementScale(0), {_0{}, _1{}, int32_t(N_gemm)}},
            {}
        },
        {
            D,
            {_1{}, int64_t{ldc}, int64_t{ldc} * int64_t{N_gemm}}
        }
    };

    typename DeviceGemm::Arguments arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N_gemm, K_gemm},
        1,
        callback_args,

        A,
        B,
        nullptr,
        nullptr,

        int64_t(M) * int64_t(K_gemm),
        int64_t(N_gemm) * int64_t(K_gemm),
        0,
        0,

        int64_t(K_gemm),  // lda: A row-major [M, K_gemm]
        int64_t(K_gemm),  // ldb: B viewed as column-major [K_gemm, N_gemm]
        0,
        0
    );

    DeviceGemm gemm_op;

    size_t workspace_size = DeviceGemm::get_workspace_size(arguments);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    cutlass::Status status;

    status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        CUTLASS_CHECK(status);
    }

    status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        CUTLASS_CHECK(status);
    }

    status = gemm_op(stream);

    /*
        Debugging helper.

        You can remove this after the kernel is stable.
        This helps catch CUTLASS errors immediately instead of later in NORM.
    */
#if 0
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUTLASS launch error: %s\n", cudaGetErrorString(err));
        if (workspace) cudaFree(workspace);
        return false;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUTLASS runtime error: %s\n", cudaGetErrorString(err));
        if (workspace) cudaFree(workspace);
        return false;
    }
#endif

    if (workspace) {
        cudaFree(workspace);
    }

    CUTLASS_CHECK(status);

    return true;
}


bool matmul_w8a8_cutlass_cuda(
    const int8_t* A,
    const int8_t* B,
    const float* alphaRow,
    const float* alphaCol,
    float* D,
    int M,
    int N_gemm,
    int K_gemm,
    int32_t ldc,
    cudaStream_t stream
) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    constexpr int kStages = 3;

    return matmul_w8a8_cutlass_f32_ptr<TileShape, WarpShape, kStages>(
        A,
        B,
        alphaRow,
        alphaCol,
        D,
        M,
        N_gemm,
        K_gemm,
        ldc,
        stream
    );
}