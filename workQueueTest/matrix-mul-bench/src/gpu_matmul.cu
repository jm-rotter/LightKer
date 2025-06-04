#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "utils.h"
#include "gpu_matmul.h"
#include <stdlib.h>
#include "lk_workqueue.h"
#include <stdio.h>
#include <iostream>


#define BLOCK_SIZE 32
/*
 * This file makes use of concepts and examples from the
 * NVIDIA CUDA C Programming Guide
 * (https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface).
 */
Matrix mA;
Matrix mB;
//Above for debugging

MatMulArgs generateMatDataPointer() {
	Matrix a = createMatrix(32, 32);
	Matrix b = createMatrix(32, 32);
	fill_random(a);
	fill_random(b);
	MatMulArgs data;
	data.A = a;
	data.B = b;
	mA = a;
	mB = b;
	return data;
}

void scheduleMatMul() {
	Input input;
	input.fn = 0;
	input.offset = 0;
 
	MatMulArgs data = generateMatDataPointer();
	int length;
	uint8_t* flattened = flatten(data, &length);
	enqueue(input, flattened, length);
}

__device__ void hhexdump(void *ptr, int buflen) {
  unsigned char *buf = (unsigned char*)ptr;
  int i, j;
  for (i=0; i<buflen; i+=16) {
    printf("%06x: ", i);
    for (j=0; j<16; j++) 
      if (i+j < buflen)
        printf("%02x ", buf[i+j]);
      else
        printf("   ");
    printf(" ");
    for (j=0; j<16; j++) 
      if (i+j < buflen)
        printf("%c", '.');
    printf("\n");
  }
}
void hexdump(void *ptr, int buflen) {
  unsigned char *buf = (unsigned char*)ptr;
  int i, j;
  for (i=0; i<buflen; i+=16) {
    printf("%06x: ", i);
    for (j=0; j<16; j++) 
      if (i+j < buflen)
        printf("%02x ", buf[i+j]);
      else
        printf("   ");
    printf(" ");
    for (j=0; j<16; j++) 
      if (i+j < buflen)
        printf("%c", isprint(buf[i+j]) ? buf[i+j] : '.');
    printf("\n");
  }
}


uint8_t* flatten(const MatMulArgs src, int* outSize) {
	log("started flatten\n");
	int lengthA = src.A.width * src.A.height;
	int lengthB = src.B.width * src.B.height;
//	printMatrix(src.A);
	uint8_t* buffer = (uint8_t*)malloc(6*sizeof(int) + (lengthA+lengthB) * sizeof(float));

	uint8_t* ptr = buffer;

	memcpy(ptr, &src.A.width, sizeof(int)); ptr += sizeof(int);
	memcpy(ptr, &src.A.height, sizeof(int)); ptr += sizeof(int);
	memcpy(ptr, &src.A.stride, sizeof(int)); ptr += sizeof(int);
	memcpy(ptr, src.A.elements, sizeof(float) * lengthA); ptr += sizeof(float) * lengthA;
	



	memcpy(ptr, &src.B.width, sizeof(int)); ptr += sizeof(int);
	memcpy(ptr, &src.B.height, sizeof(int)); ptr +=sizeof(int);
	memcpy(ptr, &src.B.stride, sizeof(int)); ptr += sizeof(int);
	memcpy(ptr, src.B.elements, sizeof(float) * lengthB); ptr += sizeof(float) * lengthB;

	*outSize = (6*sizeof(int) + (lengthA+lengthB) * sizeof(float));
//	*outSize = (3*sizeof(int) + lengthA*sizeof(float));
//	hexdump(buffer, lengthA);

	return buffer;
}

Matrix unflatten(uint8_t* src) {
	int width = *(int*)src; src+=4;
	int height = *(int*)src; src +=4;
	int stride = *(int*)src; src += 4;
	float* elements = new float[width * height];
	memcpy(elements, src, width*height*sizeof(float));

	Matrix mat;
	mat.width = width;
	mat.height = height;
	mat.stride  =stride;
	mat.elements = elements;

	return mat;

}

__device__ Matrix from_raw(int* offset) {
	printf("from raw hexdump\n");
//	hhexdump(d_arg_buffer, 8216);	
	uint8_t* ptr = d_arg_buffer + *offset;

	int width = *(int*) ptr; ptr += 4;
	int height = *(int*) ptr; ptr += 4;
	int stride = *(int*) ptr; ptr += 4;
	printf("%d, %d, %d\n", width, height, stride);
	const float* values = reinterpret_cast<const float*>(ptr);

	*offset = *offset + 12 + sizeof(float) * width*height;
	Matrix matrix;
	matrix.width = width;
	matrix.height = height;
	matrix.stride = stride;
	matrix.elements = const_cast<float*>(values);
	return matrix;
}


__device__ int naive_wrapper(int offset) {
	printf("Naive Wrapper: made it here\n");
	Matrix a = from_raw(&offset);
	printf("%d\n", sizeof(a));
	Matrix b = from_raw(&offset);
	int res_off = offset;
	printf("First two elements at %d\n", offset);

	printf("Width: %d; Height: %d; element_1: %f\n", a.width, a.height, a.elements[0]);
	printf("Width: %d; Height: %d; element_1: %f\n", b.width, b.height, b.elements[0]);

	Matrix c = from_raw(&offset);
	c.width = c.height = c.stride = 32;

	int ret =  matmul_kernel(a, b, &c);

	printf("Width: %d; Height: %d; element_1: %f\n", c.width, c.height, c.elements[0]);
	
	for(int i = 0; i < c.width*c.height; i++) {
		printf("%f;%f\n", b.elements[i], c.elements[i]);
	}

	d_input_queue[0].offset = res_off;
	d_arg_buffer[res_off] = d_arg_buffer[res_off + 4] = d_arg_buffer[res_off + 8] = 32;
	printf("%d, %d\n", d_input_queue[0].offset, res_off);
	printf("%f, \n", d_arg_buffer[res_off + 12]);

	return ret;

	//Matrix* C = (Matrix *) res;
	//int a = matmul_kernel(args->A, args->B, *C);		
	//return a;
	
}

void cpu_matmul(const Matrix& mat1, const Matrix& mat2, Matrix& res_mat) {

	for (int i = 0; i < mat1.height; i++) {
		for (int j = 0; j < mat2.width; j++) {
			float dot_product = 0.0;
			for (int k = 0; k < mat1.width; k++) {
				dot_product += mat1.elements[i * mat1.width + k] * mat2.elements[k * mat2.width + j];
			}
			res_mat.elements[i * mat2.width + j] = dot_product;
		}
	}
}

void get_result_matmul(){
	printf("Comparing Results\n");
	Input input;
	cudaError_t err = cudaMemcpyAsync(&input, diq, sizeof(Input), cudaMemcpyDeviceToHost, qStream);

	printf("%s\n", cudaGetErrorName(err));

	printf("Got the input offset: %d\n", input.offset);
	cudaStreamSynchronize(qStream);
	uint8_t* res = (uint8_t*) malloc(3*sizeof(int) + 32*32 * sizeof(float));
	printf("%d\n", input.offset);
	err = cudaMemcpyAsync(res, dab + input.offset, 3*sizeof(int) + 32*32 * sizeof(float), cudaMemcpyDeviceToHost, qStream);

	printf("%s\n", cudaGetErrorName(err));
	Matrix mres = unflatten(res);
	printMatrix(mres);
	
	Matrix res_mat = createMatrix(32, 32);
	cpu_matmul(mA, mB, res_mat);
	std::cout << "Comparison: " << (compare(res_mat, mres) ? "CPU result = GPU Naive result" : "CPU result != GPU Naive Result") << "\n";
	
}

__device__ int shared_wrapper(void* arg, void* res) { 
	MatMulArgs* args = (MatMulArgs*) arg;
	Matrix* C = (Matrix*) res;

	int a= matmul_kernel_shared_mem(args->A, args->B, *C);
	return a;
}

//Naive Implementation - One thread for each value in result_matrix
//For simplity width/height are a multiple of the block size
__device__ int matmul_kernel(const Matrix mat1, const Matrix mat2, Matrix* res_mat) {
	printf("In matrix mul kernel\n");
    float value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < mat1.width; i++) {
        value += mat1.elements[row * mat1.width + i] * mat2.elements[i * mat2.width + col];
	}

    res_mat->elements[row * res_mat->width + col] = value;
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
