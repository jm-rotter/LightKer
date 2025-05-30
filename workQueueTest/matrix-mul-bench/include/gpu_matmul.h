#include "utils.h"


typedef struct {
	Matrix A;
	Matrix B;
} MatMulArgs;

MatMulArgs generateDataPointer();
void scheduleMatMul();

//void gpu_matmul(const Matrix& mat1, const Matrix& mat2, Matrix& res_mat, GPU_imp imp);
#ifdef __CUDACC__
__global__ void matmul_kernel(const Matrix mat1, const Matrix mat2, Matrix res_mat);
__global__ void matmul_kernel_shared_mem(const Matrix mat1, const Matrix mat2, Matrix res_mat); 
__device__ float getElement(Matrix matrix, int row, int col); 
__device__ void setElement(Matrix matrix, int row, int col, float value); 
__device__ Matrix getSubMatrix(Matrix matrix, int row, int col); 
#endif

__device__ int shared_wrapper(void* arg, void* res);

__device__ int naive_wrapper(void* arg, void* res);

