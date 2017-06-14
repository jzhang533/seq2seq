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

void test_softmax() {
    int batch = 2;
    int num_labels = 4;

    Blob input;
    input.dim0 = batch;
    input.dim1 = num_labels;
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
    output.dim1 = num_labels;
    output.malloc_all_data();

    SoftmaxCompute softmax;
    softmax.init(CUDNN_SOFTMAX_ACCURATE);
    softmax.forward(&input, &output);

    output.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, batch, num_labels);
    fprintf(stderr, "output \n");
    display_matrix(output.host_data, batch, num_labels);

    // assuming a square loss
    for (int i = 0; i < output.size(); ++i) {
        output.host_diff[i] = -output.host_data[i];
    }
    output.copy_diff_to_device();

    softmax.backward(&input, &output);
    input.copy_diff_to_host();
    fprintf(stderr, "========after backward========== \n");

    fprintf(stderr, "input diff\n");
    display_matrix(input.host_diff, batch, num_labels);
    fprintf(stderr, "output diff\n");
    display_matrix(output.host_diff, batch, num_labels);
}
} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
    seq2seq::test_softmax();
    return 0;
}
