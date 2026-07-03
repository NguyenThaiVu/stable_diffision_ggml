#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

void convert_q4_0_to_int4_row_wise_cuda(
    const char * src_q4,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void quantize_fp32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

bool int4_matmul_cutlass_cuda(
    const uint8_t * A_i4_packed,
    const uint8_t * B_i4_packed,
    int32_t * C_i32,
    int M,
    int N,
    int K,
    int ldc,
    cudaStream_t stream
);

void convert_q4_0_to_int8_row_wise_cuda(
    const char * src_q4,
    int8_t * dst_i8,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void compute_smooth_alpha_q4_0_f32_cuda(
    const char * src0_q4,
    const float * src1_f32,
    float * src0_colmax,
    float * src1_colmax,
    float * smooth_alpha,
    int rows_src0,
    int rows_src1,
    int K,
    float smooth_factor,
    cudaStream_t stream
);

void convert_q4_0_to_int4_row_wise_smooth_cuda(
    const char * src_q4,
    const float * smooth_alpha,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void quantize_fp32_to_int4_row_wise_smooth_cuda(
    const float * src,
    const float * smooth_alpha,
    uint8_t * dst_i4_packed,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

void dequantize_q4_0_to_f32_cuda(
    const char * src_q4,
    float * dst_f32,
    int rows,
    int K,
    cudaStream_t stream
);

void fwht_sign_rotate_rows_cuda(
    float * x,
    int rows,
    int K,
    cudaStream_t stream
);

void quantize_f32_to_int4_row_wise_cuda(
    const float * src,
    uint8_t * dst_i4,
    float * row_scales,
    int rows,
    int K,
    cudaStream_t stream
);

bool block_fwht_sign_rotate_rows_cuda(
    float * x,
    int rows,
    int K,
    cudaStream_t stream
);

float compute_src1_incoherence_score_cuda(
    const float * src1_ddf_i,
    int64_t N,
    int64_t K,
    cudaStream_t stream
);

float get_quantization_incoherent_threshold();