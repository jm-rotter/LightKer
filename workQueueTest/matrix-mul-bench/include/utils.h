#pragma once

#include <chrono>

// Index in matrix is n row * width + j column

struct Matrix {
	int width;
	int height;
	int stride; //only relevant for gpu_shared_mem implementation
	float* elements;
};


enum GPU_imp {
	NAIVE,
	SHARED_MEMORY
};

void fill_random(Matrix& matrix);
bool compare(const Matrix& mat1, const Matrix& mat2);
double time_in_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);
Matrix createMatrix(int width, int height);
void printMatrix(Matrix matrix);
