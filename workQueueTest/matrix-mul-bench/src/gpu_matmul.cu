#pragma once
#include <cuda_runtime.h>
#include "utils.h"
#include "gpu_matmul.h"
#include <stdlib.h>
#include "lk_workqueue.h"

#define BLOCK_SIZE 32
/*
 * This file makes use of concepts and examples from the
 * NVIDIA CUDA C Programming Guide
 * (https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface).
 */


MatMulArgs generateMatDataPointer() {
	Matrix a = createMatrix(32, 32);
	Matrix b = createMatrix(32, 32);
	fill_random(a);
	fill_random(b);
	MatMulArgs data;
	data.A = a;
	data.B = b;
	return data;
}

void scheduleMatMul() {
	Task task;
	Input input;
	input.fn = 0;
 
	MatMulArgs data = generateMatDataPointer();
	input.arg = (void*) &data;
	task.input = input;
	Matrix res = createMatrix(32,32);
	task.res = (void*) &res;
	enqueue(task);
}


__device__ int naive_wrapper(void* arg, void* res) {
	MatMulArgs* args = (MatMulArgs*) args;
	Matrix* C = (Matrix *) res;
	int a = matmul_kernel(args->A, args->B, *C);		
	return a;
}

__device__ int shared_wrapper(void* arg, void* res) { 
	MatMulArgs* args = (MatMulArgs*) arg;
	Matrix* C = (Matrix*) res;

	int a= matmul_kernel_shared_mem(args->A, args->B, *C);
	return a;
}

//Naive Implementation - One thread for each value in result_matrix
//For simplity width/height are a multiple of the block size
__device__ int matmul_kernel(const Matrix mat1, const Matrix mat2, Matrix res_mat) {
	return 0;
    float value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < mat1.width; i++) {
        value += mat1.elements[row * mat1.width + i] * mat2.elements[i * mat2.width + col];
	}
    res_mat.elements[row * res_mat.width + col] = value;
	return 0;
}




//Shared Memory - One Thread for each value in result_matrix, but threads within the same block use shared_memory instead of global memory. Same process as matmul_kernel, but here we threads within a block are sharing memory when calculating each block to reduce global memory access latency.

//Device helper functions for submatrices
__device__ float getElement(Matrix matrix, int row, int col) {
	return matrix.elements[row * matrix.stride + col];
}

__device__ void setElement(Matrix matrix, int row, int col, float value) {
	matrix.elements[row * matrix.stride + col] = value;
}

__device__ Matrix getSubMatrix(Matrix matrix, int row, int col) {
	Matrix new_mat;
	new_mat.width = BLOCK_SIZE;
	new_mat.height = BLOCK_SIZE;
	new_mat.stride = matrix.stride;
	new_mat.elements = &matrix.elements[row * BLOCK_SIZE * matrix.stride + col * BLOCK_SIZE];
	return new_mat;
}

__device__ int matmul_kernel_shared_mem(const Matrix mat1, const Matrix mat2, Matrix res_mat) {
	return 0;
    	
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix subResMat = getSubMatrix(res_mat, blockRow, blockCol);

	float value = 0;  

	int row = threadIdx.y;
	int col = threadIdx.x;

	for(int i = 0; i < (mat1.width/BLOCK_SIZE); i++) {
		Matrix subMat1 = getSubMatrix(mat1, blockRow, i);
		Matrix subMat2 = getSubMatrix(mat2, i, blockCol);

		__shared__ float sMat1[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sMat2[BLOCK_SIZE][BLOCK_SIZE];

		sMat1[row][col] = getElement(subMat1, row, col);
		sMat2[row][col] = getElement(subMat2, row, col);

		__syncthreads();

		for(int j = 0; j < BLOCK_SIZE; j++) {
				value += sMat1[row][j] * sMat2[j][col];
		}

		__syncthreads();
	}
	setElement(subResMat, row, col, value);
	return 0;
}

//void gpu_matmul(const Matrix& mat1, const Matrix& mat2, Matrix& res_mat, GPU_imp imp) {
//	
//	Matrix d_mat1;
//    d_mat1.width = mat1.width; d_mat1.height = mat1.height; d_mat1.stride = mat1.stride;
//    size_t size = mat1.width * mat1.height * sizeof(float);
//    cudaMalloc(&d_mat1.elements, size);
//    cudaMemcpy(d_mat1.elements, mat1.elements, size, cudaMemcpyHostToDevice);
//
//    Matrix d_mat2;
//    d_mat2.width = mat2.width; d_mat2.height = mat2.height; d_mat2.stride = mat2.stride;
//	size = mat2.width * mat2.height * sizeof(float);
//    cudaMalloc(&d_mat2.elements, size);
//    cudaMemcpy(d_mat2.elements, mat2.elements, size, cudaMemcpyHostToDevice);
//
//	Matrix d_res_mat;
//	d_res_mat.width = res_mat.width; d_res_mat.height = res_mat.height; d_res_mat.stride = res_mat.stride;
//	size = res_mat.width * res_mat.height * sizeof(float);
//	cudaMalloc(&d_res_mat.elements, size);
//
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//	dim3 dimGrid(mat2.width / dimBlock.x, mat1.height / dimBlock.y);
//
//	switch (imp) {
//		case NAIVE:  
//			matmul_kernel<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_res_mat);
//			break;
//		case SHARED_MEMORY:
//			matmul_kernel_shared_mem<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_res_mat);
//			break;
//	}
//
//	cudaMemcpy(res_mat.elements, d_res_mat.elements, size, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_mat1.elements);
//	cudaFree(d_mat2.elements);
//	cudaFree(d_res_mat.elements);
//}
