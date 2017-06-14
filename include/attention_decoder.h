#ifndef SEQ2SEQ_INCLUDE_ATTENTION_DECODER_H
#define SEQ2SEQ_INCLUDE_ATTENTION_DECODER_H

#include "all_computes.h"

namespace seq2seq {

class AttentionDecoder {
public:
    void init(int batch_size,
            int hidden_size,
            int input_size,
            int alignment_model_size = -1,
            int maxout_size = -1,
            int max_source_seq_len = 128,
            int max_target_seq_len = 128);

    /*
     * @brief forward pass of attention decoder (assuming a bi-directional encoder)
     *
     * @param [in] input given x in shape: seq_length * batch * input_size
     * @param [in] encoder_hidden given x in shape: seq_length * batch * input_size
     * @param [out] output generate y in shape: seq_length * batch * hidden_size
     *
     * @return void
     * @retval none
     *
     */
    void forward(Blob* input,
            Blob* encoder_hidden,
            Blob* output);

    void backward(Blob* input,
            Blob* encoder_hidden,
            Blob* output);

    Blob* param_w() { return &_param_w;}
    Blob* param_u() { return &_param_u;}
    Blob* param_c() { return &_param_c;}

    Blob* param_att_w() {return &_param_att_w;};
    Blob* param_att_u() {return &_param_att_u;};
    Blob* param_att_v() {return &_param_att_v;};

    Blob* param_m_u() {return &_param_m_u;};
    Blob* param_m_v() {return &_param_m_v;};
    Blob* param_m_c() {return &_param_m_c;};

    void display_all_params();
    void display_all_params_diff();
private:
    int _batch_size;
    int _hidden_size;
    int _input_size;
    int _alignment_model_size;
    int _maxout_size;

    // will be used for preparing buffers
    int _max_source_seq_len;
    int _max_target_seq_len;

    int _source_seq_len;
    int _target_seq_len;
private:
    Blob _h0;

    Blob _param_w;  // input to hidden
    Blob _param_u;  // hidden to hidden
    Blob _param_c;  // context to hidden

    Blob _param_att_w;  // attention W
    Blob _param_att_u;  // attention U
    Blob _param_att_v;  // attention V

    Blob _param_m_u;  // maxout U
    Blob _param_m_v;  // maxout V
    Blob _param_m_c;  // maxout C

    Blob _pre_gate;   // terms before nonlinear
    Blob _gate;       // after nonlinear

    Blob _decoder_hidden;
    Blob _pre_maxout;
    Blob _max_ele_idx;

    // alignment model related
    Blob _context;           // c_i
    Blob _attention_weights; // softmax results on attention scores   alpha_ij
    Blob _attention_scores;  // before softmax   e_ij = _param_att_v *  tanh(W_a*S_i-1 + U_a*h_j)
    Blob _alignment_feats;   // tanh(W_a*S_i-1 + U_a*h_j)
    Blob _at_w_terms;      // W_a*s_i-1
    Blob _at_u_terms;      // U_a*h_j

private:
    // softmax enabled by cudnn
    cudnnTensorDescriptor_t _softmax_input_desc;
    cudnnTensorDescriptor_t _softmax_output_desc;
    cudnnSoftmaxAlgorithm_t _softmax_alg;
private:

void set_all_diff_to_zero(Blob* input, Blob* encoder_hidden);

void compute_dynamic_context(
        const Blob* encoder_hidden,
        const float* h_tm1_data,
        const int t);

void bp_dynamic_context(
        Blob* encoder_hidden,
        const float* h_data_tm1,
        float* h_diff_tm1,
        const int t);
void compute_h0_ff(Blob* encoder_hidden);
void compute_h0_bp(Blob* encoder_hidden);
};
} // namespace seq2seq

#endif
