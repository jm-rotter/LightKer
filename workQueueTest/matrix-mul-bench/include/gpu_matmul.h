#pragma once
#include "utils.h"


typedef struct {
	Matrix A;
	Matrix B;
} MatMulArgs;

MatMulArgs generateMatDataPointer();
void scheduleMatMul();

//void gpu_matmul(const Matrix& mat1, const Matrix& mat2, Matrix& res_mat, GPU_imp imp);
//#ifdef __CUDACC__
__device__ int matmul_kernel(const Matrix mat1, const Matrix mat2, Matrix res_mat);
__device__ int matmul_kernel_shared_mem(const Matrix mat1, const Matrix mat2, Matrix res_mat); 
__device__ float getElement(Matrix matrix, int row, int col); 
__device__ void setElement(Matrix matrix, int row, int col, float value); 
__device__ Matrix getSubMatrix(Matrix matrix, int row, int col); 
__device__ int shared_wrapper(void* arg, void* res);
__device__ int naive_wrapper(void* arg, void* res);
//#endif



