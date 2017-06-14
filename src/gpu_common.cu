#include "gpu_common.h"
#include <math.h>

namespace seq2seq {

#define SEQ2SEQ_TANH(x)        (__fdividef(2.0f, (1.0f + __expf(-2.0f*(x)))) - 1.0f)
#define SEQ2SEQ_TANH_D(x)      (1.0f - (x) * (x))
#define SEQ2SEQ_SIGMOID(x)     (__fdividef(1.0f, 1.0f + __expf(-(x))))
#define SEQ2SEQ_SIGMOID_D(x)   ((x) * (1.0f - (x)))


__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;

   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

#define EMB_BATCH_THREADS_X    32
#define EMB_BATCH_BLOCKS_X     4
#define EMB_BATCH_BLOCKS_Y     128

__global__ void emb_ff_kernel(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size) {
    int itx = threadIdx.y * blockDim.x + threadIdx.x;
    int ity = blockIdx.x;
    for (int batchid = blockIdx.y; batchid < batch_size; batchid += gridDim.y) {
        //DTYPE *dst_t = output + top_offset[batchid] * len;
        float *dst_t = output + batchid * seq_length * emb_size;
        //const DTYPE *index_t = word + offset[batchid];
        const float* index_t = input + batchid * seq_length;
        for (int j = ity; j < seq_length; j += gridDim.x) {
            const float* emb_t = w + static_cast<unsigned int>(index_t[j]) * emb_size;
            float *dst_x = dst_t + j * emb_size;
            for (int i = itx; i < emb_size; i += blockDim.x * blockDim.y) {
                dst_x[i] = emb_t[i];
            }
        }
    }
}

void emb_ff(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size) {
    dim3 blocks(EMB_BATCH_BLOCKS_X, EMB_BATCH_BLOCKS_Y);
    dim3 threads(EMB_BATCH_THREADS_X, 1);
    emb_ff_kernel<<< blocks, threads >>> (w, input, output, batch_size, seq_length, emb_size);
}

__global__ void emb_bp_kernel(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr) {
    int itx = threadIdx.y * blockDim.x + threadIdx.x;

    for (int batchid = blockIdx.y; batchid < batch_size; batchid += gridDim.y) {
        const float* word = input + batchid * seq_length;
        for (int ity = blockIdx.x; ity < seq_length; ity += gridDim.x) {
            const float* grad_t = grad_output + (batchid * seq_length + ity) * emb_size;
            float* dst_t = w + static_cast<unsigned int>(word[ity]) * emb_size;
            for (int i = itx; i < emb_size; i += blockDim.x * blockDim.y) {
                atomicAdd(dst_t + i,  mlr * grad_t[i]);
            }
        }
    }
}

void emb_bp(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr) {
    dim3 blocks(EMB_BATCH_BLOCKS_X, EMB_BATCH_BLOCKS_Y);
    dim3 threads(EMB_BATCH_THREADS_X, 1);

    emb_bp_kernel<<< blocks, threads >>> (w,
            input,
            grad_output,
            batch_size,
            seq_length,
            emb_size,
            mlr);
}


//////////////////////////////////////////////////////
// embedding ff/bp for feeding to rnn compute
// ff result shape is seq_length * batch * emb_size
/////////////////////////////////////////////////////

__global__ void emb_ff_for_rnn_kernel(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size) {
    int total = seq_length * batch_size * emb_size;
    CUDA_KERNEL_LOOP(i, total) {
        int row = i / emb_size;
        int column = i % emb_size;

        const float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
        output[i] = emb_t[column];
    }
}

void emb_ff_for_rnn(const float* w,
        const float* input,
        float* output,
        int batch_size,
        int seq_length,
        int emb_size) {
    int total = seq_length * batch_size * emb_size;
    const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
    const dim3 gridSize(GET_BLOCKS(total), 1, 1);
    emb_ff_for_rnn_kernel<<< gridSize, blockSize >>> (w, input, output, batch_size, seq_length, emb_size);
}

__global__ void emb_bp_for_rnn_kernel(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr) {
    int total = seq_length * batch_size * emb_size;
    CUDA_KERNEL_LOOP(i, total) {
        int row = i / emb_size;
        int column = i % emb_size;

        float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
        atomicAdd(emb_t + column,  mlr * grad_output[i]);
    }
}

void emb_bp_for_rnn(float* w,
        const float* input,
        const float* grad_output,
        int batch_size,
        int seq_length,
        int emb_size,
        const float mlr) {
    int total = seq_length * batch_size * emb_size;
    emb_bp_for_rnn_kernel<<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(
            w,
            input,
            grad_output,
            batch_size,
            seq_length,
            emb_size,
            mlr);
}

__global__ void negative_loss_ff_kernel(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        int pad_id) {
    CUDA_KERNEL_LOOP(i, batch) {
        unsigned int true_label = static_cast<unsigned int>(labels[i]);
        if (true_label != pad_id) {
            output[i] = -input[i * num_labels + true_label];
        } else {
            output[i] = 0.0;
        }
    }
}

// TODO: return the result of real examples (not pad_id)
void negative_loss_ff(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        int pad_id) {
    negative_loss_ff_kernel<<<GET_BLOCKS(batch), CUDA_NUM_THREADS>>>(
            input,
            labels,
            output,
            batch,
            num_labels,
            pad_id);
}

__global__ void negative_loss_bp_kernel(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        float loss_factor,
        int pad_id) {
    CUDA_KERNEL_LOOP(i, batch * num_labels) {
        unsigned int batch_id = i / num_labels;
        unsigned int this_label = i % num_labels;
        unsigned int true_label = static_cast<unsigned int>(labels[batch_id]);

        if (true_label == pad_id || this_label != true_label) {
            output[i] = 0.0;
        } else {
            output[i] = -loss_factor;
        }
    }
}

void negative_loss_bp(
        const float* input,
        const float* labels,
        float* output,
        int batch,
        int num_labels,
        float loss_factor,
        int pad_id) {
    negative_loss_bp_kernel<<<GET_BLOCKS(batch * num_labels), CUDA_NUM_THREADS>>>(
            input,
            labels,
            output,
            batch,
            num_labels,
            loss_factor,
            pad_id);
}

__global__ void add_at_w_and_u_terms_and_nonlinear_kernel(
        const float* w_terms,
        const float* u_terms,
        float* alignment_feats,
        int seq_len,
        int batch_size,
        int alignment_model_size) {

    CUDA_KERNEL_LOOP(i, seq_len * batch_size * alignment_model_size) {
        unsigned int col_id = i % (batch_size * alignment_model_size);
        alignment_feats[i] = SEQ2SEQ_TANH(w_terms[col_id] + u_terms[i]);
    }
}

void add_at_w_and_u_terms_and_nonlinear(
        const float* w_terms,
        const float* u_terms,
        float* alignment_feats,
        int seq_len,
        int batch_size,
        int alignment_model_size) {
    add_at_w_and_u_terms_and_nonlinear_kernel<<<GET_BLOCKS(seq_len * batch_size * alignment_model_size), CUDA_NUM_THREADS>>>(
            w_terms,
            u_terms,
            alignment_feats,
            seq_len,
            batch_size,
            alignment_model_size);
}

__global__ void add_at_w_and_u_terms_and_nonlinear_bp_kernel(
        const float* alignment_feats,
        const float* alignment_feats_diff,
        float* w_terms_diff,
        float* u_terms_diff,
        int seq_len,
        int batch_size,
        int alignment_model_size) {
    CUDA_KERNEL_LOOP(i, seq_len * batch_size * alignment_model_size) {
        unsigned int col_id = i % (batch_size * alignment_model_size);
        float tanhd = SEQ2SEQ_TANH_D(alignment_feats[i]) * alignment_feats_diff[i];
        u_terms_diff[i] += tanhd;
        atomicAdd(w_terms_diff + col_id, tanhd);
    }
    // TODO: avoid atomicAdd
}

void add_at_w_and_u_terms_and_nonlinear_bp(
        const float* alignment_feats,
        const float* alignment_feats_diff,
        float* w_terms_diff,
        float* u_terms_diff,
        int seq_len,
        int batch_size,
        int alignment_model_size) {
    add_at_w_and_u_terms_and_nonlinear_bp_kernel<<<GET_BLOCKS(seq_len * batch_size * alignment_model_size), CUDA_NUM_THREADS>>>(
            alignment_feats,
            alignment_feats_diff,
            w_terms_diff,
            u_terms_diff,
            seq_len,
            batch_size,
            alignment_model_size);
}


__global__ void compute_context_kernel(
        const float* attention_weights,
        const float* encoder_hidden,
        float* context,
        int seq_len,
        int batch_size,
        int hidden_size) {

    CUDA_KERNEL_LOOP(i, batch_size * 2 * hidden_size) {
        int batch_id = i / (2 * hidden_size);
        context[i] = 0.0;
        for (int k = 0; k < seq_len; ++k) {
            context[i] += encoder_hidden[k * batch_size * 2 * hidden_size + i] \
                    * attention_weights[k * batch_size + batch_id];
        }
    }
}

void compute_context(
        const float* attention_weights,
        const float* encoder_hidden,
        float* context,
        int seq_len,
        int batch_size,
        int hidden_size) {

    compute_context_kernel<<<GET_BLOCKS(batch_size * 2 * hidden_size), CUDA_NUM_THREADS>>>(
            attention_weights,
            encoder_hidden,
            context,
            seq_len,
            batch_size,
            hidden_size);

}
__global__ void bp_compute_context_kernel(
        const float* context_diff,
        const float* attention_weights,
        const float* encoder_hidden,
        float* attention_weights_diff,
        float* encoder_hidden_diff,
        int seq_len,
        int batch_size,
        int hidden_size) {
    CUDA_KERNEL_LOOP(i, seq_len * batch_size * 2 * hidden_size) {
        int j = i / (2 * hidden_size);
        int k = i % (batch_size * 2 * hidden_size);
        atomicAdd(attention_weights_diff + j,  encoder_hidden[i] * context_diff[k]);
        // NOTICE, use += here, since every step in decoder has this diff
        encoder_hidden_diff[i] += attention_weights[j] * context_diff[k];
        //atomicAdd(encoder_hidden_diff + i, attention_weights[j] * context_diff[k]);
    }
    // TODO: use a reduce paradigm to avoid atomicAdd
}

void bp_compute_context(
        const float* context_diff,
        const float* attention_weights,
        const float* encoder_hidden,
        float* attention_weights_diff,
        float* encoder_hidden_diff,
        int seq_len,
        int batch_size,
        int hidden_size) {
    // CAUTION HERE: only memset attention weights diff, dont memset hidden_diff
//    cudaMemset(encoder_hidden_diff, 0.0, sizeof(float) * seq_len * batch_size * 2 * hidden_size);
    cudaMemset(attention_weights_diff, 0.0, sizeof(float) * seq_len * batch_size);

    bp_compute_context_kernel<<<GET_BLOCKS(seq_len * batch_size * 2 * hidden_size), CUDA_NUM_THREADS>>>(
            context_diff,
            attention_weights,
            encoder_hidden,
            attention_weights_diff,
            encoder_hidden_diff,
            seq_len,
            batch_size,
            hidden_size);
}

__global__ void attention_decoder_ff_nonlinear_kernel(
        const float* h_data_tm1,
        const float* pre_gate_data_w_t,
        const float* pre_gate_data_u_t,
        const float* pre_gate_data_c_t,
        float* gate_data_t,
        float* h_data_t,
        const int batch_size,
        const int hidden_size) {
    CUDA_KERNEL_LOOP(i, batch_size * hidden_size) {
        int batch_id = i / hidden_size;
        int k = i % hidden_size;

        // reset gate
        int r_idx = batch_id * 3 * hidden_size + k;
        gate_data_t[r_idx] = SEQ2SEQ_SIGMOID(pre_gate_data_w_t[r_idx] \
                + pre_gate_data_u_t[r_idx]
                + pre_gate_data_c_t[r_idx]);

        // update gate
        int z_idx = (batch_id * 3 + 1) * hidden_size + k;
        gate_data_t[z_idx] = SEQ2SEQ_SIGMOID(pre_gate_data_w_t[z_idx] \
                + pre_gate_data_u_t[z_idx] \
                + pre_gate_data_c_t[z_idx]);

        // new gate
        int n_idx = (batch_id * 3 + 2 ) * hidden_size + k;
        gate_data_t[n_idx] = SEQ2SEQ_TANH(pre_gate_data_w_t[n_idx] \
                + gate_data_t[r_idx] * pre_gate_data_u_t[n_idx] \
                + pre_gate_data_c_t[n_idx]);

        // output
        h_data_t[i] = (1.0 -  gate_data_t[z_idx]) * h_data_tm1[i] + gate_data_t[z_idx] * gate_data_t[n_idx];
    }
}


void attention_decoder_ff_nonlinear(
        const float* h_data_tm1,
        const float* pre_gate_data_w_t,
        const float* pre_gate_data_u_t,
        const float* pre_gate_data_c_t,
        float* gate_data_t,
        float* h_data_t,
        const int batch_size,
        const int hidden_size) {
    attention_decoder_ff_nonlinear_kernel<<<GET_BLOCKS(batch_size * hidden_size), CUDA_NUM_THREADS>>>(
            h_data_tm1,
            pre_gate_data_w_t,
            pre_gate_data_u_t,
            pre_gate_data_c_t,
            gate_data_t,
            h_data_t,
            batch_size,
            hidden_size);
}


__global__ void attention_decoder_bp_nonlinear_kernel(
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
        const int hidden_size) {
    CUDA_KERNEL_LOOP(i, batch_size * hidden_size) {
        int batch_id = i / hidden_size;
        int k = i % hidden_size;

        // reset gate index
        int r_idx = batch_id * 3 * hidden_size + k;
        // update gate index
        int z_idx = (batch_id * 3 + 1) * hidden_size + k;
        // new gate index
        int n_idx = (batch_id * 3 + 2 ) * hidden_size + k;

        // grads wrt h_tm1, using += since it already has diff from upper computation
        h_diff_tm1[i] += (1.0 - gate_data_t[z_idx]) * h_diff_t[i];

        // grads wrt new gate
        gate_diff_t[n_idx] = gate_data_t[z_idx] * h_diff_t[i];
        // nonlinear grads
        float n_grad = gate_diff_t[n_idx] * SEQ2SEQ_TANH_D(gate_data_t[n_idx]);
        pre_gate_diff_w_t[n_idx] = n_grad;
        pre_gate_diff_u_t[n_idx] = n_grad * gate_data_t[r_idx];
        pre_gate_diff_c_t[n_idx] = n_grad;

        // grads wrt update gate
        gate_diff_t[z_idx] = (gate_data_t[n_idx] - h_data_tm1[i]) * h_diff_t[i];
        // nonlinear grads
        float z_grad = gate_diff_t[z_idx] * SEQ2SEQ_SIGMOID_D(gate_data_t[z_idx]);
        pre_gate_diff_w_t[z_idx] = z_grad;
        pre_gate_diff_u_t[z_idx] = z_grad;
        pre_gate_diff_c_t[z_idx] = z_grad;

        // grads wrt reset gate
        gate_diff_t[r_idx] = n_grad * pre_gate_data_u_t[n_idx];
        float r_grad = gate_diff_t[r_idx] * SEQ2SEQ_SIGMOID_D(gate_data_t[r_idx]);
        pre_gate_diff_w_t[r_idx] = r_grad;
        pre_gate_diff_u_t[r_idx] = r_grad;
        pre_gate_diff_c_t[r_idx] = r_grad;
    }
}

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
        const int hidden_size) {
    attention_decoder_bp_nonlinear_kernel<<<GET_BLOCKS(batch_size * hidden_size), CUDA_NUM_THREADS>>>(
            h_data_tm1,
            h_diff_t,
            gate_data_t,
            pre_gate_data_u_t,
            h_diff_tm1,
            pre_gate_diff_w_t,
            pre_gate_diff_u_t,
            pre_gate_diff_c_t,
            gate_diff_t,
            batch_size,
            hidden_size);
}

__global__ void copy_for_decoder_h0_data_kernel(
        const float* encoder_hidden_data,
        float* h0_data,
        int batch_size,
        int hidden_size) {
    CUDA_KERNEL_LOOP(i, batch_size * hidden_size) {
        int batch_id = i / hidden_size;
        int k = i % hidden_size;
        h0_data[i] = encoder_hidden_data[2 * hidden_size * batch_id + hidden_size + k];
    }
}
void copy_for_decoder_h0_data(
        const float* encoder_hidden_data,
        float* h0_data,
        int batch_size,
        int hidden_size) {
    copy_for_decoder_h0_data_kernel<<<GET_BLOCKS(batch_size * hidden_size), CUDA_NUM_THREADS>>>(
            encoder_hidden_data,
            h0_data,
            batch_size,
            hidden_size);
}
__global__ void copy_for_decoder_h0_diff_kernel(
        const float* h0_diff,
        float* encoder_hidden_diff,
        int batch_size,
        int hidden_size) {
    CUDA_KERNEL_LOOP(i, batch_size * hidden_size) {
        int batch_id = i / hidden_size;
        int k = i % hidden_size;
        // use += here, since it already has diff
        encoder_hidden_diff[2 * hidden_size * batch_id + hidden_size + k] += h0_diff[i];
    }
}

void copy_for_decoder_h0_diff(
        const float* h0_diff,
        float* encoder_hidden_diff,
        int batch_size,
        int hidden_size) {
    copy_for_decoder_h0_diff_kernel<<<GET_BLOCKS(batch_size * hidden_size), CUDA_NUM_THREADS>>>(
            h0_diff,
            encoder_hidden_diff,
            batch_size,
            hidden_size);
}

__global__ void maxout_ff_kernel(
        const float* pre_maxout_data,
        float* maxout_data,
        float* maxout_ele_idx,
        int total_output_size) {
    CUDA_KERNEL_LOOP(i, total_output_size) {
        int k = 2 * i;
        int kp1 = 2 * i + 1;
        if (pre_maxout_data[k] > pre_maxout_data[kp1]) {
            maxout_data[i] = pre_maxout_data[k];
            maxout_ele_idx[i] = k;
        } else {
            maxout_data[i] = pre_maxout_data[kp1];
            maxout_ele_idx[i] = kp1;
        }
    }
}

void maxout_ff(
        const float* pre_maxout_data,
        float* maxout_data,
        float* maxout_ele_idx,
        int total_output_size) {
    maxout_ff_kernel<<<GET_BLOCKS(total_output_size), CUDA_NUM_THREADS>>>(
            pre_maxout_data,
            maxout_data,
            maxout_ele_idx,
            total_output_size);
}

__global__ void maxout_bp_kernel(
        float* pre_maxout_diff,
        const float* maxout_diff,
        const float* maxout_ele_idx,
        int total_output_size) {
    CUDA_KERNEL_LOOP(i, total_output_size) {
        int idx = static_cast<int>(maxout_ele_idx[i]);
        pre_maxout_diff[idx] = maxout_diff[i];
    }
}

void maxout_bp(
        float* pre_maxout_diff,
        const float* maxout_diff,
        const float* maxout_ele_idx,
        int total_output_size) {
    maxout_bp_kernel<<<GET_BLOCKS(total_output_size), CUDA_NUM_THREADS>>>(
            pre_maxout_diff,
            maxout_diff,
            maxout_ele_idx,
            total_output_size);
}
} // namespace seq2seq
