#include "cpu_matmul.h"
#include "utils.h"


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
