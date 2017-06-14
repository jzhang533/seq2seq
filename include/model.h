#ifndef SEQ2SEQ_INCLUDE_MODEL_H
#define SEQ2SEQ_INCLUDE_MODEL_H

#include "data_reader.h"
#include "all_computes.h"
#include "attention_decoder.h"

namespace seq2seq {

// for simplicity, make all members public
// TODO: refactoring it
// TODO: use sampled softmax to reduce gpu memory
struct Seq2SeqModel {
    float lr;
    int batch_size;
    int emb_size;
    int hidden_size;
    int alignment_size;
    int maxout_size;
    int source_voc_size;
    int target_voc_size;

    void init(int max_encoder_seq_len, int max_decoder_seq_len);
    float forward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
    void backward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
    void clip_gradients(float max_gradient_norm);
    void optimize(Blob* encoder_input, Blob* decoder_input);

    // load model
    void load_model(const char* dirname);
    // save model into this directory
    void save_model(const char* dirname);
    EmbCompute encoder_emb;
    EmbCompute decoder_emb;
    RNNCompute encoder_rnn;
    AttentionDecoder decoder_rnn;
    FCCompute fc_compute;
    SoftmaxCompute softmax;
    NegativeLossCompute loss_compute;

    Blob encoder_embedding_blob;
    Blob decoder_embedding_blob;

    Blob encoder_rnn_blob;
    Blob encoder_rnn_final_hidden;

    Blob decoder_rnn_blob;

    Blob presoftmax_blob;
    Blob softmax_result_blob;

    Blob loss_blob;

private:
    float loss_factor;
};

} // namespace seq2seq
#endif
