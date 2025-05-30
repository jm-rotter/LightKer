#include <cmath>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include "utils.h"

void fill_random(Matrix& matrix) {
	for(int i = 0; i < matrix.height * matrix.width; i++) {
		matrix.elements[i] = static_cast<float>(rand()) / RAND_MAX;
	}
}

bool compare(const Matrix& mat1, const Matrix& mat2) {
	if(mat1.width != mat2.width || mat1.height != mat2.height) {
		std::cout << "!ERROR: Incorrect width and height" << "\n";
		return false;
	}
	for(int i = 0; i < mat1.width * mat1.height; i++) {
		if(fabs(mat1.elements[i] - mat2.elements[i]) > 1e-3) {
			std::cout << "!ERROR: Incorrect Values " << mat1.elements[i] << " vs " << mat2.elements[i] << "\n";
			return false;
		}
	}
	return true;
}
double time_in_ms(std::chrono::high_resolution_clock::time_point start, 
		          std::chrono::high_resolution_clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

Matrix createMatrix(int width, int height) {
	Matrix mat;
	mat.width = width;
	mat.stride = width;
	mat.height = height;
	mat.elements = new float[width * height];
	return mat;
}

