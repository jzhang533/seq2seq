#ifndef SEQ2SEQ_INCLUDE_COMMON_H
#define SEQ2SEQ_INCLUDE_COMMON_H

#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "mkl.h"
#include "cudnn.h"
#include "gpu_common.h"

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
TypeName(TypeName&) = delete;              \
void operator=(TypeName) = delete;

#define cudaErrCheck(condition) \
do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

#define cublasErrCheck(condition) \
do { \
    cublasStatus_t status = condition; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cublas Error: error code : %d %s %d\n", status, __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

#define cudnnErrCheck(condition) \
do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error: %s %s %d\n", \
                cudnnGetErrorString(status), \
                __FILE__, \
                __LINE__); \
        exit(-1); \
    } \
} while (0)

namespace seq2seq {
class GlobalAssets {
public:
    explicit GlobalAssets() {}
    // TODO: and deconstructor to close the handles
    static GlobalAssets* instance();

    inline cublasHandle_t& cublasHandle() {
        return _cublasHandle;
    }

    inline cudnnHandle_t& cudnnHandle() {
        return _cudnnHandle;
    }

private:
static std::shared_ptr<GlobalAssets> _s_singleton_global_assets;
cublasHandle_t _cublasHandle;
cudnnHandle_t _cudnnHandle;
private:
DISALLOW_COPY_AND_ASSIGN(GlobalAssets);
};


void cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);

void gpu_gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C);

void gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y);

float uniform_rand(float min, float max);
void xavier_fill(float* data, int count, int in, int out);
void constant_fill(float* data, int count, float val);

template <typename Dtype>
void display_matrix(const Dtype* data, int row, int col, int dim2 = -1);

static const std::string space_string = std::string(" ");
void split(const std::string& main_str,
        std::vector<std::string>& str_list,
        const std::string& delimiter = space_string);

} // namespace seq2seq
#endif
