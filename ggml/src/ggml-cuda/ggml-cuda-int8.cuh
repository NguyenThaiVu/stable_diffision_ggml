#pragma once

/*
 * Custom INT8 CUDA helpers for ggml experiments.
 *
 * This header should contain declarations only.
 * Function bodies should stay in ggml-cuda-int8.cu.
 */

#include "ggml.h"

#include <cuda_runtime.h>
#include <stdint.h>

/* During the running, we call
`GGML_USE_CUSTOM_KERNEL=1 ./my_program`
to enable the custom INT8 cublas implementation.
*/
bool use_custom_kernel();

void print_ggml_tensor_info(
    const struct ggml_tensor * t,
    const char * name
);

// ================================================================
// Row-wise FP32 -> INT8 quantization
// input:  [rows, cols], row-major
// output: [rows, cols], row-major
// scales: [rows]
// ================================================================

void quantize_fp32_to_int8_row_wise_cuda(
    const float * input,
    int8_t * output,
    float * scales,
    int rows,
    int cols,
    cudaStream_t stream
);

// ================================================================
// Column-wise FP32 -> INT8 quantization
// input:  [rows, cols], column-major
// output: [rows, cols], column-major
// scales: [cols]
// ================================================================

void quantize_fp32_to_int8_col_wise_cuda(
    const float * input,
    int8_t * output,
    float * scales,
    int rows,
    int cols,
    cudaStream_t stream
);

// ================================================================
// INT32 -> FP32 dequantization
// input_i32:  [rows, cols], column-major with leading dimension ldc
// output_f32: [rows, cols], column-major with leading dimension ldc
// row_scales: [rows]
// col_scales: [cols]
// ================================================================

void dequantize_i32_to_f32_cuda(
    const int32_t * input_i32,
    float * output_f32,
    const float * row_scales,
    const float * col_scales,
    int rows,
    int cols,
    int ldc,
    cudaStream_t stream
);


// ================================================================
void convert_q8_0_to_int8_row_wise_cuda(
    const void * input_q8_0,
    int8_t * output_i8,
    float * output_scales,
    int rows,
    int cols,
    cudaStream_t stream
);


// ===============================================================
bool int8_matmul_cutlass_cuda(const int8_t* input,
    const int8_t* weight,
    int32_t* output,
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,
    cudaStream_t stream
);

// ===============================================================
bool matmul_w8a8_cutlass_cuda(
    const int8_t* A,                  // [M, K_gemm], row-major
    const int8_t* B,                  // [N_gemm, K_gemm], row-major physical memory
    const float* alphaRow,            // [M]
    const float* alphaCol,            // [N_gemm]
    float* D,                         // [M, N_gemm], column-major physical memory
    int M,
    int N_gemm,
    int K_gemm,
    int ldc,
    cudaStream_t stream
);

