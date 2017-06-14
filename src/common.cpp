#include "common.h"

namespace seq2seq {
std::shared_ptr<GlobalAssets> GlobalAssets::_s_singleton_global_assets;

GlobalAssets* GlobalAssets::instance() {
    if (_s_singleton_global_assets.get() == NULL) { // first call, create the instance
        _s_singleton_global_assets.reset(new GlobalAssets());
        cublasErrCheck(cublasCreate(&_s_singleton_global_assets->cublasHandle()));
        cudnnErrCheck(cudnnCreate(&_s_singleton_global_assets->cudnnHandle()));
    }
    return _s_singleton_global_assets.get();
}

void cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

void gpu_gemm(const CBLAS_TRANSPOSE TransA,
  const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
  const float alpha, const float* A, const float* B, const float beta,
  float* C) {
    // Note that cublas follows fortran order.
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

//    fprintf(stderr, "n=%d m=%d k=%d lda=%d ldb=%d\n", N, M, K, lda, ldb);
    cublasErrCheck(
      cublasSgemm(GlobalAssets::instance()->cublasHandle(), cuTransB, cuTransA,
          N, M, K, &alpha, B, ldb, A, lda, &beta, C, N)
    );
}

void gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
    cublasOperation_t cuTransA =
            (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasErrCheck(cublasSgemv(GlobalAssets::instance()->cublasHandle(),
            cuTransA, N, M, &alpha,
            A, N, x, 1, &beta, y, 1));
}

float uniform_rand(float min, float max) {
    return ((float) rand() / (RAND_MAX)) * (max - min) + min;
}

void xavier_fill(float* data, int count, int in, int out) {
    float scale = sqrt(float(6.0) / (in + out));
    for (int i = 0; i < count; ++i) {
        data[i] = uniform_rand(-scale, scale);
    }
}

void constant_fill(float* data, int count, float val) {
    for (int i = 0; i < count; ++i) {
        data[i] = val;
    }
}

void display_matrix_helper(const float* data, int row, int col) {
    for (int i = 0; i < row; ++i) {
        char buffer[102400];
        *buffer = 0;
        for (int j = 0; j < col; ++j) {
            int len = strlen(buffer);
            snprintf(buffer + len,
                    102400 - len,
                    "%+.6f%s",
                    data[i * col + j],
                    j == col - 1 ? "" : ", "
                    );
        }
        fprintf(stderr, "[%s]%s", buffer, i == row - 1 ? "" : "\n");
    }
}

void display_matrix_helper(const int* data, int row, int col) {
    for (int i = 0; i < row; ++i) {
        char buffer[102400];
        *buffer = 0;
        for (int j = 0; j < col; ++j) {
            int len = strlen(buffer);
            snprintf(buffer + len,
                    102400 - len,
                    "%d%s",
                    data[i * col + j],
                    j == col - 1 ? "" : ", "
                    );
        }
        fprintf(stderr, "[%s]%s", buffer, i == row - 1 ? "" : "\n");
    }
}

template <typename Dtype>
void display_matrix(const Dtype* data, int row, int col, int dim2/* = -1*/) {
    if (dim2 != -1) {
        fprintf(stderr, "maxtrix at mem:%p, %d, %d, %d\n", data, row, col, dim2);
        for (int i = 0; i < row; ++i) {
            fprintf(stderr, "[");
            display_matrix_helper(data + i * col * dim2, col, dim2);
            fprintf(stderr, "]\n\n");
        }
        return;
    } else {
        fprintf(stderr, "maxtrix at mem:%p, %d, %d\n", data, row, col);
        display_matrix_helper(data, row, col);
        fprintf(stderr, "\n");
    }
}
template
void display_matrix<float>(const float* data, int row, int col, int dim2/* = -1*/);

template
void display_matrix<int>(const int* data, int row, int col, int dim2/* = -1*/);

void split(const std::string& main_str,
        std::vector<std::string>& str_list,
        const std::string& delimiter /* = space_string */) {
    size_t pre_pos = 0;
    size_t position = 0;
    std::string tmp_str;

    str_list.clear();
    if (main_str.empty()) {
        return;
    }

    while ((position = main_str.find(delimiter, pre_pos)) != std::string::npos) {
        tmp_str.assign(main_str, pre_pos, position - pre_pos);

        // notice: empty string will also be counted
        // e.g.: when delimiter is ";", ";;" will also have an empty string
        // if (!tmp_str.empty())
        {
            str_list.push_back(tmp_str);
        }
        pre_pos = position + 1;
    }

    tmp_str.assign(main_str, pre_pos, main_str.length() - pre_pos);

    if (!tmp_str.empty()) {
        str_list.push_back(tmp_str);
    }
}
} // namespace seq2seq
