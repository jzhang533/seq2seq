#include <cstring>
#include <unistd.h>
#include <assert.h>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <cuda.h>
#include "all_computes.h"
#include "attention_decoder.h"

namespace seq2seq {

void test_attention_decoder() {
    int batch = 2;
    int source_seq_len = 5;
    int target_seq_len = 3;
    int input_size = 4;
    int hidden_size = 6;
    int maxout_size = 8;
    int alignment_model_size = 10;

    //////////////// input
    Blob input;
    input.dim0 = target_seq_len;
    input.dim1 = batch;
    input.dim2 = input_size;
    input.malloc_all_data();
    float data[input.size()];

    for (int i = 0; i < input.size(); ++i) {
        data[i] = uniform_rand(-0.1, 0.1);
    }
    memcpy(input.host_data, data, input.size() * sizeof(float));
    input.copy_data_to_device();

    //////////////// encoder_hidden
    Blob encoder_hidden;
    encoder_hidden.dim0 = source_seq_len;
    encoder_hidden.dim1 = batch;
    encoder_hidden.dim2 = 2 * hidden_size;
    encoder_hidden.malloc_all_data();
    float encoder_data[encoder_hidden.size()];

    for (int i = 0; i < encoder_hidden.size(); ++i) {
        encoder_data[i] = uniform_rand(-0.1, 0.1);
    }
    memcpy(encoder_hidden.host_data, encoder_data, encoder_hidden.size() * sizeof(float));
    encoder_hidden.copy_data_to_device();

    //////////////// output
    Blob output;
    output.dim0 = target_seq_len;
    output.dim1 = batch;
    output.dim2 = maxout_size;
    output.malloc_all_data();

    AttentionDecoder decoder;
    decoder.init(batch,
            hidden_size,
            input_size,
            alignment_model_size,
            maxout_size,
            source_seq_len,
            target_seq_len);
#if 1
    FILE* fp = fopen("/tmp/input.bin", "wb");
    fwrite(input.host_data, input.size() * sizeof(float), 1, fp);
    fclose(fp);
    fp = fopen("/tmp/encoder_hidden.bin", "wb");
    fwrite(encoder_hidden.host_data, encoder_hidden.size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_w.bin", "wb");
    fwrite(decoder.param_w()->host_data, decoder.param_w()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_u.bin", "wb");
    fwrite(decoder.param_u()->host_data, decoder.param_u()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_c.bin", "wb");
    fwrite(decoder.param_c()->host_data, decoder.param_c()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_att_w.bin", "wb");
    fwrite(decoder.param_att_w()->host_data, decoder.param_att_w()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_att_u.bin", "wb");
    fwrite(decoder.param_att_u()->host_data, decoder.param_att_u()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_att_v.bin", "wb");
    fwrite(decoder.param_att_v()->host_data, decoder.param_att_v()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_m_u.bin", "wb");
    fwrite(decoder.param_m_u()->host_data, decoder.param_m_u()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_m_v.bin", "wb");
    fwrite(decoder.param_m_v()->host_data, decoder.param_m_v()->size() * sizeof(float), 1, fp);
    fclose(fp);
 
    fp = fopen("/tmp/decoder.param_m_c.bin", "wb");
    fwrite(decoder.param_m_c()->host_data, decoder.param_m_c()->size() * sizeof(float), 1, fp);
    fclose(fp);
#endif

    decoder.display_all_params();

    // do more than one bp to make sure internal buffers are cleand
    static const int times_of_bp = 1;
    for (int k = 0; k < times_of_bp; ++k) {
        fprintf(stderr, "========round %d forward========== \n", k);
        fprintf(stderr, "input \n");
        display_matrix(input.host_data, target_seq_len, batch,  input_size);

        fprintf(stderr, "encoder_hidden \n");
        display_matrix(encoder_hidden.host_data, source_seq_len, batch,  2 * hidden_size);

        decoder.forward(&input, &encoder_hidden, &output);

        fprintf(stderr, "output \n");
        output.copy_data_to_host();
        display_matrix(output.host_data, target_seq_len, batch, maxout_size);

    //    return;

        // a squared loss is given
        for (int i = 0; i < output.size(); ++i) {
            output.host_diff[i] = -output.host_data[i];
        }
        output.copy_diff_to_device();

        fprintf(stderr, "output diff\n");
        display_matrix(output.host_diff, target_seq_len, batch, hidden_size);

        decoder.backward(&input, &encoder_hidden, &output);

        input.copy_diff_to_host();
        encoder_hidden.copy_diff_to_host();
        output.copy_diff_to_host();

        fprintf(stderr, "========round %d after backward========== \n", k);
        fprintf(stderr, "output diff\n");
        display_matrix(output.host_diff, target_seq_len, batch, maxout_size);

        fprintf(stderr, "input diff\n");
        display_matrix(input.host_diff, target_seq_len, batch, input_size);
        fprintf(stderr, "encoder_hidden diff\n");
        display_matrix(encoder_hidden.host_diff, source_seq_len, batch, 2 * hidden_size);
        fprintf(stderr, "weights diff\n");
        // TODO: display weights diff
        decoder.display_all_params_diff();
    }
}

} // namespace seq2seq

int main(int argc, char** argv) {
//    srand(time(NULL));
    seq2seq::test_attention_decoder();
    return 0;
}
