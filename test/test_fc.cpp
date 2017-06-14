#include <cstring>
#include <unistd.h>
#include <assert.h>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <cuda.h>
#include "all_computes.h"

namespace seq2seq {
void test_math() {
    int batch = 2;
    int in = 3;
    int out = 4;

    float data[batch * in];
    data[0] = data[1] = data[2] = 1.0;
    data[3] = data[4] = data[5] = 10.0;
    float* data_device = NULL;
    cudaErrCheck(cudaMalloc((void**)&data_device, batch * in * sizeof(float)));
    cudaErrCheck(cudaMemcpy(data_device,
          data,
          batch * in * sizeof(float),
          cudaMemcpyHostToDevice));

    float w[in * out];
    for (int i = 0; i < in * out; ++i) {
        w[i] = 1;
    }
    float* w_device = NULL;
    cudaErrCheck(cudaMalloc((void**)&w_device, in * out * sizeof(float)));
    cudaErrCheck(cudaMemcpy(w_device,
          w,
          in * out * sizeof(float),
          cudaMemcpyHostToDevice));

    float output[batch * out];
    for (int i = 0; i < batch * out; ++i) {
        output[i] = 0;
    }
    float* output_device = NULL;
    cudaErrCheck(cudaMalloc((void**)&output_device, batch * out * sizeof(float)));
    cudaErrCheck(cudaMemcpy(output_device,
          output,
          batch * out * sizeof(float),
          cudaMemcpyHostToDevice));

#if 0
        cpu_gemm(CblasNoTrans,
                CblasNoTrans,
                batch,
                out,
                in,
                1.0,
                data,
                w,
                0.0,
                output);
#else
        gpu_gemm(CblasNoTrans,
                CblasNoTrans,
                batch,
                out,
                in,
                1.0,
                data_device,
                w_device,
                0.0,
                output_device);
        cudaErrCheck(cudaMemcpy(output,
                output_device,
                batch * out * sizeof(float),
                cudaMemcpyDeviceToHost));
#endif

    display_matrix(output, batch, out);
}


void test_fc() {
    int batch = 6;
    int in = 4;
    int out = 6;

    Blob input;
    input.dim0 = batch;
    input.dim1 = in;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = uniform_rand(-0.1, 0.1);
//        data[i] = 1;
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    Blob output;
    output.dim0 = batch;
    output.dim1 = out;
    output.malloc_all_data();

    FCCompute fc;
    fc.init(in, out);

#if 1
    FILE* fp = fopen("/tmp/input.bin", "wb");
    fwrite(input.host_data, input.size() * sizeof(float), 1, fp);
    fclose(fp);
    fp = fopen("/tmp/weights.bin", "wb");
    fc.get_w()->copy_data_to_host();
    fwrite(fc.get_w()->host_data, fc.get_w()->size() * sizeof(float), 1, fp);
    fclose(fp);
#endif

    fc.forward(&input, &output);

    output.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, batch, in);
    fprintf(stderr, "weights \n");
    display_matrix(fc.get_w()->host_data, in, out);
    fprintf(stderr, "bias \n");
    display_matrix(fc.get_b()->host_data, 1, out);
    fprintf(stderr, "output \n");
    display_matrix(output.host_data, batch, out);

    for (int i = 0; i < output.size(); ++i) {
//        output.host_diff[i] = uniform_rand(-0.1, 0.1);
//        output.host_diff[i] = 1.0f;
        output.host_diff[i] = -output.host_data[i];
    }
    output.copy_diff_to_device();

    for (int k = 0; k < 4; ++k) {
        fc.backward(&input, &output);
        fc.get_w()->copy_diff_to_host();
        fc.get_b()->copy_diff_to_host();
        input.copy_diff_to_host();
        fprintf(stderr, "========after backward========== \n");

        fprintf(stderr, "input diff\n");
        display_matrix(input.host_diff, batch, in);
        fprintf(stderr, "output diff\n");
        display_matrix(output.host_diff, batch, out);
        fprintf(stderr, "weights diff\n");
        display_matrix(fc.get_w()->host_diff, in, out);
        fprintf(stderr, "bias diff\n");
        display_matrix(fc.get_b()->host_diff, 1, out);
    }
}
} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
    seq2seq::test_math();
    seq2seq::test_fc();
    return 0;
}
