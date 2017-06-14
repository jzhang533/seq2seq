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

void test_emb() {
    int batch = 2;
    int voc_size = 10;
    int emb_size = 2;
    int seq_length = 5;

    Blob input;
    input.dim0 = batch;
    input.dim1 = seq_length;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = rand() % voc_size;
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    Blob output;
    output.dim0 = batch;
    output.dim1 = seq_length;
    output.dim2 = emb_size;
    output.malloc_all_data();

    EmbCompute emb_compute;
    emb_compute.init(voc_size, emb_size);
//    for (int i = 0; i< 1000000; ++i) {
    emb_compute.forward(&input, &output);
//   }

    output.copy_data_to_host();

    fprintf(stderr, "input \n");
    display_matrix(input.host_data, batch, seq_length);
    fprintf(stderr, "weights \n");
    display_matrix(emb_compute.get_w()->host_data, voc_size, emb_size);
    fprintf(stderr, "output \n");
    display_matrix(output.host_data, batch, seq_length * emb_size);

    cudaErrCheck(cudaMemset(output.device_diff, 0, output.size() * sizeof(float)));
    initGPUData(output.device_diff, output.size(), 1.0f);
    output.copy_diff_to_host();
    fprintf(stderr, "output diff\n");
    display_matrix(output.host_diff, batch, seq_length * emb_size);

    emb_compute.backward(&input, &output);
    emb_compute.get_w()->copy_data_to_host();

    fprintf(stderr, "==========after backward========= \n");
    fprintf(stderr, "input \n");
    display_matrix(input.host_data, batch, seq_length);
    fprintf(stderr, "weights \n");
    display_matrix(emb_compute.get_w()->host_data, voc_size, emb_size);
    fprintf(stderr, "output \n");
    display_matrix(output.host_data, batch, seq_length * emb_size);
}

void test_emb_for_rnn() {
    int batch = 2;
    int voc_size = 10;
    int emb_size = 2;
    int seq_length = 5;

    Blob input;
    input.dim0 = seq_length;
    input.dim1 = batch;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = rand() % voc_size;
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    Blob output;
    output.dim0 = seq_length;
    output.dim1 = batch;
    output.dim2 = emb_size;
    output.malloc_all_data();

    EmbCompute emb_compute;
    emb_compute.init(voc_size, emb_size);
//    for (int i = 0; i< 1000000; ++i) {
    emb_compute.forward_for_rnn(&input, &output);
//   }

    output.copy_data_to_host();

    fprintf(stderr, "input \n");
    input.display_data();
    fprintf(stderr, "weights \n");
    emb_compute.get_w()->display_data();
    fprintf(stderr, "output \n");
    output.display_data();

    cudaErrCheck(cudaMemset(output.device_diff, 0, output.size() * sizeof(float)));
    initGPUData(output.device_diff, output.size(), 1.0f);
    output.copy_diff_to_host();
    fprintf(stderr, "output diff\n");
    output.display_diff();

    emb_compute.set_lr(0.1);
    emb_compute.backward_for_rnn(&input, &output);
    emb_compute.get_w()->copy_data_to_host();

    fprintf(stderr, "==========after backward========= \n");
    fprintf(stderr, "input \n");
    input.display_data();
    fprintf(stderr, "weights \n");
    emb_compute.get_w()->display_data();
    fprintf(stderr, "output \n");
    output.display_data();
}
} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
//    seq2seq::test_emb();
//    fprintf(stderr, "==========================\n");
    seq2seq::test_emb_for_rnn();
    return 0;
}
