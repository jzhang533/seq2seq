#include "data_reader.h"

namespace seq2seq {
void test_load_dict() {
    DataReader reader;
    reader.load_source_dict("nist-data-debug/source.vocab");
    reader.load_target_dict("nist-data-debug/target.vocab");

    fprintf(stderr, "source dict size : %d, target dict size: %d\n",
            reader.source_dict_size(),
            reader.target_dict_size());

    static const int batch_size = 4;
    Blob encoder_input;
    Blob decoder_input;
    Blob decoder_target;

    encoder_input.dim0 = 50;
    encoder_input.dim1 = batch_size;
    decoder_input.dim0 = 52;
    decoder_input.dim1 = batch_size;
    decoder_target.dim0 = 52;
    decoder_target.dim1 = batch_size;

    encoder_input.malloc_all_data();
    decoder_input.malloc_all_data();
    decoder_target.malloc_all_data();

    reader.load_all_data(
            "nist-data-debug/source",
            "nist-data-debug/target",
            batch_size,
            50,
            50);

    for (int i = 0; i < 1000; ++i) {
        fprintf(stderr, "========iter %d==========\n", i);
        reader.get_batch(&encoder_input, &decoder_input, &decoder_target);
        display_matrix(encoder_input.host_data, encoder_input.dim0, encoder_input.dim1);
        display_matrix(decoder_input.host_data, decoder_input.dim0, decoder_input.dim1);
        display_matrix(decoder_target.host_data, decoder_target.dim0, decoder_target.dim1);
        //reader.display_all_data();
    }
}
} // namespace seq2seq

int main(int argc, char** argv) {
    srand(time(NULL));
    seq2seq::test_load_dict();
    return 0;
}
