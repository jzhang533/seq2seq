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

void display_rnn_params(RNNCompute& rnn) {
    for (size_t i = 0; i < rnn.get_matrix_blobs().size(); ++i) {
        for (size_t j = 0; j < rnn.get_matrix_blobs()[i].size(); ++j) {
            Blob& matrix = rnn.get_matrix_blobs()[i][j];
            matrix.copy_data_to_host();
            display_matrix(matrix.host_data, matrix.dim0, matrix.dim1);
        }
    }

    for (size_t i = 0; i < rnn.get_bias_blobs().size(); ++i) {
        for (size_t j = 0; j < rnn.get_bias_blobs()[i].size(); ++j) {
            Blob& bias = rnn.get_bias_blobs()[i][j];
            bias.copy_data_to_host();
            display_matrix(bias.host_data, bias.dim0, bias.dim1);
        }
    }
}

void display_rnn_dw(RNNCompute& rnn) {
    size_t weights_size = rnn.weights_size();
    float host_data[weights_size];
    cudaErrCheck(cudaMemcpy(host_data,
            rnn.get_dw(),
            weights_size,
            cudaMemcpyDeviceToHost));

    display_matrix(host_data, 1, weights_size / sizeof(float));
}

void test_rnn() {
    int batch = 2;
    int seq_length = 3;
    int input_size = 4;
    int hidden_size = 6;

    Blob input;
    input.dim0 = seq_length;
    input.dim1 = batch;
    input.dim2 = input_size;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = uniform_rand(-0.1, 0.1);
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    Blob output;
    output.dim0 = seq_length;
    output.dim1 = batch;
    output.dim2 = hidden_size;
    output.malloc_all_data();

    Blob final_hidden;
    final_hidden.dim0 = batch;
    final_hidden.dim1 = hidden_size;
    final_hidden.malloc_all_data();

    RNNCompute rnn;
    rnn.init(batch,
            hidden_size,
            input_size,
            true,
            1,
            false,
            CUDNN_GRU,
            0.0);

#if 1
    FILE* fp = fopen("/tmp/input.bin", "wb");
    fwrite(input.host_data, input.size() * sizeof(float), 1, fp);
    fclose(fp);
    fp = fopen("/tmp/weights.bin", "wb");
    rnn.get_param_blob()->copy_data_to_host();
    fwrite(rnn.get_param_blob()->host_data, rnn.get_param_blob()->size() * sizeof(float), 1, fp);
    fclose(fp);
#endif

    rnn.forward(&input, &output, NULL, NULL, NULL, NULL);

    output.copy_data_to_host();
    final_hidden.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, seq_length, batch * input_size);

    fprintf(stderr, "param \n");
    display_rnn_params(rnn);

    fprintf(stderr, "output \n");
    display_matrix(output.host_data, seq_length, batch * hidden_size);

    fprintf(stderr, "final hidden \n");
    display_matrix(final_hidden.host_data, batch, hidden_size);

#if 1
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < output.size(); ++i) {
            output.host_diff[i] = -output.host_data[i];
        }
        output.copy_diff_to_device();

        rnn.backward(&input, &output);
        //    fc.get_w()->copy_diff_to_host();
        //    fc.get_b()->copy_diff_to_host();
        input.copy_diff_to_host();
        fprintf(stderr, "========after backward========== \n");

        fprintf(stderr, "input diff\n");
        display_matrix(input.host_diff, seq_length, batch * input_size);
        fprintf(stderr, "output diff\n");
        display_matrix(output.host_diff, seq_length, batch * hidden_size);
        fprintf(stderr, "weights diff\n");
        display_rnn_dw(rnn);
    }
#endif
}

void test_birnn() {
    int batch = 2;
    int seq_length = 3;
    int input_size = 4;
    int hidden_size = 6;

    Blob input;
    input.dim0 = seq_length;
    input.dim1 = batch;
    input.dim2 = input_size;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = uniform_rand(-0.1, 0.1);
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    Blob output;
    output.dim0 = seq_length;
    output.dim1 = batch;
    output.dim2 = 2 * hidden_size;
    output.malloc_all_data();

    Blob final_hidden;
    final_hidden.dim0 = batch;
    final_hidden.dim1 = 2 * hidden_size;
    final_hidden.malloc_all_data();

    RNNCompute rnn;
    rnn.init(batch,
            hidden_size,
            input_size,
            true,
            1,
            true,
            CUDNN_GRU,
            0.0);

    rnn.forward(&input, &output, NULL, NULL, &final_hidden, NULL);

    output.copy_data_to_host();
    final_hidden.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, seq_length, batch * input_size);

    fprintf(stderr, "param \n");
    display_rnn_params(rnn);

    fprintf(stderr, "output \n");
    display_matrix(output.host_data, seq_length, batch * 2 * hidden_size);

    fprintf(stderr, "final hidden \n");
    display_matrix(final_hidden.host_data, batch, 2 * hidden_size);

#if 0
    for (int i = 0; i < output.size(); ++i) {
//        output.host_diff[i] = uniform_rand(-0.1, 0.1);
//        output.host_diff[i] = 1.0f;
        output.host_diff[i] = -output.host_data[i];
    }
    output.copy_diff_to_device();
    for (int k = 0; k < 4; ++k) {
        rnn.backward(&input, &output);
        //    fc.get_w()->copy_diff_to_host();
        //    fc.get_b()->copy_diff_to_host();
        input.copy_diff_to_host();
        fprintf(stderr, "========after backward========== \n");

        fprintf(stderr, "input diff\n");
        display_matrix(input.host_diff, seq_length, batch * input_size);
        fprintf(stderr, "output diff\n");
        display_matrix(output.host_diff, seq_length, batch * hidden_size);
        fprintf(stderr, "weights diff\n");
        display_rnn_dw(rnn);
    }
#endif
}

} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
    seq2seq::test_rnn();
//    seq2seq::test_birnn();
    return 0;
}
