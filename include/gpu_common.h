#ifndef SEQ2SEQ_INCLUDE_GPU_COMMON_H
#define SEQ2SEQ_INCLUDE_GPU_COMMON_H

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace seq2seq {

const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void initGPUData(float *data, int numElements, float value);
void emb_ff(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size);

// this is actually a update of embedding matrix
// needs to redesign in multi device implementation
void emb_bp(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr);

// for feeding to rnn
void emb_ff_for_rnn(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size);

void emb_bp_for_rnn(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr);

void negative_loss_ff(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        int pad_id);

void negative_loss_bp(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        float loss_factor,
        int pad_id);

void add_at_w_and_u_terms_and_nonlinear(
        const float* w_terms,
        const float* u_terms,
        float* alignment_feats,
        int seq_len,
        int batch_size,
        int alignment_model_size);

void add_at_w_and_u_terms_and_nonlinear_bp(
        const float* alignment_feats,
        const float* alignment_feats_diff,
        float* w_terms_diff,
        float* u_terms_diff,
        int seq_len,
        int batch_size,
        int alignment_model_size);

void compute_context(
        const float* attention_weights,
        const float* encoder_hidden,
        float* context,
        int seq_len,
        int batch_size,
        int hidden_size);

void bp_compute_context(
        const float* context_diff,
        const float* attention_weights,
        const float* encoder_hidden,
        float* attention_weights_diff,
        float* encoder_hidden_diff,
        int seq_len,
        int batch_size,
        int hidden_size);

void attention_decoder_ff_nonlinear(
        const float* h_data_tm1,
        const float* pre_gate_data_w_t,
        const float* pre_gate_data_u_t,
        const float* pre_gate_data_c_t,
        float* gate_data_t,
        float* h_data_t,
        const int batch_size,
        const int hidden_size);

void attention_decoder_bp_nonlinear(
        const float* h_data_tm1,
        const float* h_diff_t,
        const float* gate_data_t,
        const float* pre_gate_data_u_t,
        float* h_diff_tm1,
        float* pre_gate_diff_w_t,
        float* pre_gate_diff_u_t,
        float* pre_gate_diff_c_t,
        float* gate_diff_t,
        const int batch_size,
        const int hidden_size);

void copy_for_decoder_h0_data(
        const float* encoder_hidden_data,
        float* h0_data,
        int batch_size,
        int hidden_size);
void copy_for_decoder_h0_diff(
        const float* h0_diff,
        float* encoder_hidden_diff,
        int batch_size,
        int hidden_size);

void maxout_ff(
        const float* pre_maxout_data,
        float* maxout_data,
        float* maxout_ele_idx,
        int total_output_size);

void maxout_bp(
        float* pre_maxout_diff,
        const float* maxout_diff,
        const float* maxout_ele_idx,
        int total_output_size);

} // namespace seq2seq

#endif
