#include "cpu_matmul.h"
#include "gpu_matmul.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <iostream>
#include <vector>
#include <chrono>


int main() {

	int a, b, c;

	//std::cout << "Please type in mat1 height and width and mat2 width (mat2 height = mat1 width)\n";
	//std::cin >> a >> b >> c;

	a = 1024;
	b = 1024;
	c = 1024;

	std::cout << "Matrix Multiplicatoin Benchmark with 1024 x 1024 Matrices \n";

	Matrix mat1 = createMatrix(a, b);
	Matrix mat2 = createMatrix(b, c);
	Matrix gpuNaiveRes = createMatrix(a, c);
	Matrix gpuSharedRes = createMatrix(a, c);
	Matrix cpuRes = createMatrix(a, c);

	fill_random(mat1);
	fill_random(mat2);

	auto t1 = std::chrono::high_resolution_clock::now();
	cpu_matmul(mat1, mat2, cpuRes);
	auto t2 = std::chrono::high_resolution_clock::now();
	double cpuTime = time_in_ms(t1, t2);
	std::cout << "CPU Time: " << cpuTime << " ms\n";



	t1 = std::chrono::high_resolution_clock::now();
	gpu_matmul(mat1,mat2, gpuNaiveRes, NAIVE);
	cudaDeviceSynchronize();
	t2 = std::chrono::high_resolution_clock::now();
	double gpuNaiveTime = time_in_ms(t1, t2);
	std::cout << "GPU Naive Implementation Time: " << gpuNaiveTime << " ms\n";


	t1 = std::chrono::high_resolution_clock::now();
	gpu_matmul(mat1,mat2, gpuSharedRes, SHARED_MEMORY);
	cudaDeviceSynchronize();
	t2 = std::chrono::high_resolution_clock::now();
	double gpuSharedTime = time_in_ms(t1, t2);
	std::cout << "GPU Shared Memory Implementation Time: " << gpuSharedTime << " ms\n";



	std::cout << "Comparison: " << (compare(gpuNaiveRes, cpuRes) ? "CPU result = GPU Naive result" : "CPU result != GPU Naive Result") << "\n";


	std::cout << "Comparison: " << (compare(gpuNaiveRes, gpuSharedRes) ? "GPU Naive result = GPU Shared Memory result" : "GPU Naive result != GPU Shared Memory Result") << "\n";


	std::cout << "Speedup CPU -> GPU Naive: " << cpuTime / gpuNaiveTime << "\n";

	std::cout << "Speedup GPU Naive -> GPU Shared Memory: " << gpuNaiveTime / gpuSharedTime << "\n";

}
