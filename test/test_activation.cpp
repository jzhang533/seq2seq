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

void test_activation() {
    int batch = 2;
    int num = 4;

    Blob input;
    input.dim0 = batch;
    input.dim1 = num;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = uniform_rand(-0.1, 0.1);
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

#if 0 // use inplace or not
    Blob& output = input;
#else
    Blob output;
    output.dim0 = batch;
    output.dim1 = num;
    output.malloc_all_data();
#endif

    ActivationCompute activation;
    // CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH
    activation.init(CUDNN_ACTIVATION_SIGMOID);
    activation.forward(&input, &output);

    output.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, batch, num);
    fprintf(stderr, "output \n");
    display_matrix(output.host_data, batch, num);

    // assuming a square loss
    for (int i = 0; i < output.size(); ++i) {
        output.host_diff[i] = -output.host_data[i];
    }
    output.copy_diff_to_device();

    activation.backward(&input, &output);
    input.copy_diff_to_host();
    fprintf(stderr, "========after backward========== \n");

    fprintf(stderr, "input diff\n");
    display_matrix(input.host_diff, batch, num);
    fprintf(stderr, "output diff\n");
    display_matrix(output.host_diff, batch, num);
}
} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
    seq2seq::test_activation();
    return 0;
}
