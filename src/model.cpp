#include "model.h"

namespace seq2seq {

// load it into text format
void Seq2SeqModel::load_model(const char* dirname) {
    std::string encoder_emb_weights_filename = std::string(dirname) + "/encoder.emb";
    encoder_emb.get_w()->loadtxt(encoder_emb_weights_filename.c_str());

    std::string decoder_emb_weights_filename = std::string(dirname) + "/decoder.emb";
    decoder_emb.get_w()->loadtxt(decoder_emb_weights_filename.c_str());

    std::string encoder_rnn_weights_filename = std::string(dirname) + "/encoder_rnn.weights";
    encoder_rnn.get_param_blob()->loadtxt(encoder_rnn_weights_filename.c_str());

    std::string decoder_rnn_w_filename = std::string(dirname) + "/decoder_rnn.weights.w";
    decoder_rnn.param_w()->loadtxt(decoder_rnn_w_filename.c_str());
    std::string decoder_rnn_u_filename = std::string(dirname) + "/decoder_rnn.weights.u";
    decoder_rnn.param_u()->loadtxt(decoder_rnn_u_filename.c_str());
    std::string decoder_rnn_c_filename = std::string(dirname) + "/decoder_rnn.weights.c";
    decoder_rnn.param_c()->loadtxt(decoder_rnn_c_filename.c_str());

    std::string decoder_rnn_att_w_filename = std::string(dirname) + "/decoder_rnn.weights.att_w";
    decoder_rnn.param_att_w()->loadtxt(decoder_rnn_att_w_filename.c_str());
    std::string decoder_rnn_att_u_filename = std::string(dirname) + "/decoder_rnn.weights.att_u";
    decoder_rnn.param_att_u()->loadtxt(decoder_rnn_att_u_filename.c_str());
    std::string decoder_rnn_att_v_filename = std::string(dirname) + "/decoder_rnn.weights.att_v";
    decoder_rnn.param_att_v()->loadtxt(decoder_rnn_att_v_filename.c_str());

    std::string decoder_rnn_m_u_filename = std::string(dirname) + "/decoder_rnn.weights.m_u";
    decoder_rnn.param_m_u()->loadtxt(decoder_rnn_m_u_filename.c_str());
    std::string decoder_rnn_m_v_filename = std::string(dirname) + "/decoder_rnn.weights.m_v";
    decoder_rnn.param_m_v()->loadtxt(decoder_rnn_m_v_filename.c_str());
    std::string decoder_rnn_m_c_filename = std::string(dirname) + "/decoder_rnn.weights.m_c";
    decoder_rnn.param_m_c()->loadtxt(decoder_rnn_m_c_filename.c_str());

    std::string fc_weights_filename = std::string(dirname) + "/fc.weights";
    fc_compute.get_w()->loadtxt(fc_weights_filename.c_str());

    std::string fc_bias_filename = std::string(dirname) + "/fc.bias";
    fc_compute.get_b()->loadtxt(fc_bias_filename.c_str());
}

// save it into text format
void Seq2SeqModel::save_model(const char* dirname) {
    std::string encoder_emb_weights_filename = std::string(dirname) + "/encoder.emb";
    encoder_emb.get_w()->savetxt(encoder_emb_weights_filename.c_str());

    std::string decoder_emb_weights_filename = std::string(dirname) + "/decoder.emb";
    decoder_emb.get_w()->savetxt(decoder_emb_weights_filename.c_str());

    std::string encoder_rnn_weights_filename = std::string(dirname) + "/encoder_rnn.weights";
    encoder_rnn.get_param_blob()->savetxt(encoder_rnn_weights_filename.c_str());

    std::string decoder_rnn_w_filename = std::string(dirname) + "/decoder_rnn.weights.w";
    decoder_rnn.param_w()->savetxt(decoder_rnn_w_filename.c_str());
    std::string decoder_rnn_u_filename = std::string(dirname) + "/decoder_rnn.weights.u";
    decoder_rnn.param_u()->savetxt(decoder_rnn_u_filename.c_str());
    std::string decoder_rnn_c_filename = std::string(dirname) + "/decoder_rnn.weights.c";
    decoder_rnn.param_c()->savetxt(decoder_rnn_c_filename.c_str());

    std::string decoder_rnn_att_w_filename = std::string(dirname) + "/decoder_rnn.weights.att_w";
    decoder_rnn.param_att_w()->savetxt(decoder_rnn_att_w_filename.c_str());
    std::string decoder_rnn_att_u_filename = std::string(dirname) + "/decoder_rnn.weights.att_u";
    decoder_rnn.param_att_u()->savetxt(decoder_rnn_att_u_filename.c_str());
    std::string decoder_rnn_att_v_filename = std::string(dirname) + "/decoder_rnn.weights.att_v";
    decoder_rnn.param_att_v()->savetxt(decoder_rnn_att_v_filename.c_str());

    std::string decoder_rnn_m_u_filename = std::string(dirname) + "/decoder_rnn.weights.m_u";
    decoder_rnn.param_m_u()->savetxt(decoder_rnn_m_u_filename.c_str());
    std::string decoder_rnn_m_v_filename = std::string(dirname) + "/decoder_rnn.weights.m_v";
    decoder_rnn.param_m_v()->savetxt(decoder_rnn_m_v_filename.c_str());
    std::string decoder_rnn_m_c_filename = std::string(dirname) + "/decoder_rnn.weights.m_c";
    decoder_rnn.param_m_c()->savetxt(decoder_rnn_m_c_filename.c_str());

    std::string fc_weights_filename = std::string(dirname) + "/fc.weights";
    fc_compute.get_w()->savetxt(fc_weights_filename.c_str());

    std::string fc_bias_filename = std::string(dirname) + "/fc.bias";
    fc_compute.get_b()->savetxt(fc_bias_filename.c_str());
}

void Seq2SeqModel::clip_gradients(float max_gradient_norm) {
    float fc_sumsq = 0.0;
    float encoder_rnn_sumsq = 0.0;
    float decoder_rnn_sumsq = 0.0;

    cublasErrCheck(cublasSdot(
            GlobalAssets::instance()->cublasHandle(),
            fc_compute.get_w()->size(),
            fc_compute.get_w()->device_diff,
            1,
            fc_compute.get_w()->device_diff,
            1,
            &fc_sumsq));

    cublasErrCheck(cublasSdot(
            GlobalAssets::instance()->cublasHandle(),
            encoder_rnn.weights_size() / sizeof(float),
            encoder_rnn.get_dw(),
            1,
            encoder_rnn.get_dw(),
            1,
            &encoder_rnn_sumsq));

    std::vector<Blob*> rnn_decoder_param_blobs;
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_w());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_c());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_w());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_v());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_v());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_c());

    for (size_t i = 0; i < rnn_decoder_param_blobs.size(); ++i) {
        float temp_sumsq = 0.0;
        cublasErrCheck(cublasSdot(
                GlobalAssets::instance()->cublasHandle(),
                rnn_decoder_param_blobs[i]->size(),
                rnn_decoder_param_blobs[i]->device_diff,
                1,
                rnn_decoder_param_blobs[i]->device_diff,
                1,
                &temp_sumsq));
        decoder_rnn_sumsq += temp_sumsq;
    }

    float all_sumsq = fc_sumsq + encoder_rnn_sumsq + decoder_rnn_sumsq;
    float global_norm = sqrt(all_sumsq);

#if 0
    fprintf(stderr, "encoder length %d decoder length %d\n", encoder_rnn_blob.dim0, decoder_rnn_blob.dim0);
    float fc_bottom_diff = 0.0;
    cublasErrCheck(cublasSdot(
            GlobalAssets::instance()->cublasHandle(),
            decoder_rnn_blob.size(),
            decoder_rnn_blob.device_diff,
            1,
            decoder_rnn_blob.device_diff,
            1,
            &fc_bottom_diff));

    fprintf(stderr, "decoder_fc bottom: %.8f\n", fc_bottom_diff);

    fprintf(stderr, "global norm %.6f fc_sumsq %.6f encoder_rnn_sumsq %.6f decoder_rnn_sumsq %.6f \n",
            global_norm,
            fc_sumsq,
            encoder_rnn_sumsq,
            decoder_rnn_sumsq);
#endif

    if (global_norm > max_gradient_norm) {
        fprintf(stderr, "global norm %.6f exceeds threshold %.6f, clipping gradients\n",
                global_norm,
                max_gradient_norm);

        float scale_factor = max_gradient_norm / global_norm;

        cublasErrCheck(cublasSscal(
                GlobalAssets::instance()->cublasHandle(),
                fc_compute.get_w()->size(),
                &scale_factor,
                fc_compute.get_w()->device_diff,
                1));

        cublasErrCheck(cublasSscal(
                GlobalAssets::instance()->cublasHandle(),
                encoder_rnn.weights_size() / sizeof(float),
                &scale_factor,
                encoder_rnn.get_dw(),
                1));

        for (size_t i = 0; i < rnn_decoder_param_blobs.size(); ++i) {
            cublasErrCheck(cublasSscal(
                    GlobalAssets::instance()->cublasHandle(),
                    rnn_decoder_param_blobs[i]->size(),
                    &scale_factor,
                    rnn_decoder_param_blobs[i]->device_diff,
                    1));
        }

        // althoug not count in embedding parameters when calculating global norm
        // also needs to scale embedding grads
        cublasErrCheck(cublasSscal(
                GlobalAssets::instance()->cublasHandle(),
                encoder_embedding_blob.size(),
                &scale_factor,
                encoder_embedding_blob.device_diff,
                1));
        cublasErrCheck(cublasSscal(
                GlobalAssets::instance()->cublasHandle(),
                decoder_embedding_blob.size(),
                &scale_factor,
                decoder_embedding_blob.device_diff,
                1));
    }
}

void Seq2SeqModel::optimize(Blob* encoder_input, Blob* decoder_input) {
    float mlr = -lr;

    cublasErrCheck(cublasSaxpy(
            GlobalAssets::instance()->cublasHandle(),
            fc_compute.get_w()->size(),
            &mlr,
            fc_compute.get_w()->device_diff,
            1,
            fc_compute.get_w()->device_data,
            1));

    cublasErrCheck(cublasSaxpy(
            GlobalAssets::instance()->cublasHandle(),
            encoder_rnn.weights_size() / sizeof(float),
            &mlr,
            encoder_rnn.get_dw(),
            1,
            encoder_rnn.get_param_blob()->device_data,
            1));


    std::vector<Blob*> rnn_decoder_param_blobs;
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_w());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_c());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_w());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_att_v());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_u());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_v());
    rnn_decoder_param_blobs.push_back(decoder_rnn.param_m_c());

    for (size_t i = 0; i < rnn_decoder_param_blobs.size(); ++i) {
    cublasErrCheck(cublasSaxpy(
            GlobalAssets::instance()->cublasHandle(),
            rnn_decoder_param_blobs[i]->size(),
            &mlr,
            rnn_decoder_param_blobs[i]->device_diff,
            1,
            rnn_decoder_param_blobs[i]->device_data,
            1));
    }

    // set learning rate for embedding compute since its being updated during backward pass
    decoder_emb.set_lr(lr);
    encoder_emb.set_lr(lr);

    decoder_emb.backward_for_rnn(decoder_input, &decoder_embedding_blob);
    encoder_emb.backward_for_rnn(encoder_input, &encoder_embedding_blob);
}

void Seq2SeqModel::init(int max_encoder_seq_len, int max_decoder_seq_len) {
    alignment_size = hidden_size;
    maxout_size = hidden_size;

    encoder_emb.init(source_voc_size, emb_size);
    decoder_emb.init(target_voc_size, emb_size);
    encoder_rnn.init(batch_size,
            hidden_size,
            emb_size,
            true,
            1,
            true,
            CUDNN_GRU,
            0.0);

    decoder_rnn.init(batch_size,
            hidden_size,
            emb_size,
            alignment_size,
            maxout_size,
            max_encoder_seq_len,
            max_decoder_seq_len);

    fc_compute.init(maxout_size, target_voc_size);

    softmax.init(CUDNN_SOFTMAX_LOG);
    loss_compute.init(DataReader::PAD_ID);

    // prepare intermedia blobs
    encoder_embedding_blob.dim0 = max_encoder_seq_len;
    encoder_embedding_blob.dim1 = batch_size;
    encoder_embedding_blob.dim2 = emb_size;
    encoder_embedding_blob.malloc_all_data();

    decoder_embedding_blob.dim0 = max_decoder_seq_len;
    decoder_embedding_blob.dim1 = batch_size;
    decoder_embedding_blob.dim2 = emb_size;
    decoder_embedding_blob.malloc_all_data();

    encoder_rnn_blob.dim0 = max_encoder_seq_len;
    encoder_rnn_blob.dim1 = batch_size;
    encoder_rnn_blob.dim2 = 2 * hidden_size;
    encoder_rnn_blob.malloc_all_data();

    encoder_rnn_final_hidden.dim0 = batch_size;
    encoder_rnn_final_hidden.dim1 = 2 * hidden_size;
    encoder_rnn_final_hidden.malloc_all_data();

    decoder_rnn_blob.dim0 = max_decoder_seq_len;
    decoder_rnn_blob.dim1 = batch_size;
    decoder_rnn_blob.dim2 = hidden_size;
    decoder_rnn_blob.malloc_all_data();

    presoftmax_blob.dim0 = max_decoder_seq_len * batch_size;
    presoftmax_blob.dim1 = target_voc_size;
    presoftmax_blob.malloc_all_data();

    softmax_result_blob.dim0 = max_decoder_seq_len * batch_size;
    softmax_result_blob.dim1 = target_voc_size;
    softmax_result_blob.malloc_all_data();

    loss_blob.dim0 = max_decoder_seq_len * batch_size;
    loss_blob.dim1 = 1;
    loss_blob.malloc_all_data();
}

float Seq2SeqModel::forward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target) {
    // seq_len * batch * emb_size
    encoder_embedding_blob.dim0 = encoder_input->dim0;
    encoder_embedding_blob.dim1 = encoder_input->dim1;
    encoder_embedding_blob.dim2 = emb_size;
    encoder_emb.forward_for_rnn(encoder_input, &encoder_embedding_blob);

    // seq_len * batch * emb_size
    decoder_embedding_blob.dim0 = decoder_input->dim0;
    decoder_embedding_blob.dim1 = decoder_input->dim1;
    decoder_embedding_blob.dim2 = emb_size;
    decoder_emb.forward_for_rnn(decoder_input, &decoder_embedding_blob);

    // seq_len * batch *  2 * hidden_size
    encoder_rnn_blob.dim0 = encoder_embedding_blob.dim0;
    encoder_rnn_blob.dim1 = encoder_embedding_blob.dim1;
    encoder_rnn_blob.dim2 = 2 * hidden_size;

    // batch * 2 * hidden_size
    encoder_rnn_final_hidden.dim0 = encoder_embedding_blob.dim1;
    encoder_rnn_final_hidden.dim1 = 2 * hidden_size;
    encoder_rnn_final_hidden.dim2 = 1;
    encoder_rnn.forward(&encoder_embedding_blob,
            &encoder_rnn_blob,
            NULL,
            NULL,
            NULL);

//    encoder_input->display_data("input");
//    encoder_rnn_blob.display_data("encoder matrix");

    // seq_len * batch * hidden_size
    decoder_rnn_blob.dim0 = decoder_embedding_blob.dim0;
    decoder_rnn_blob.dim1 = decoder_embedding_blob.dim1;
    decoder_rnn_blob.dim2 = hidden_size;
    decoder_rnn.forward(&decoder_embedding_blob,
            &encoder_rnn_blob,
            &decoder_rnn_blob);

//    decoder_rnn_blob.display_data("decoder hidden");

    // before fc, reshpae the input to (seq_len * batch) * hidden_size
    decoder_rnn_blob.dim0 = decoder_rnn_blob.dim0 * decoder_rnn_blob.dim1;
    decoder_rnn_blob.dim1 = decoder_rnn_blob.dim2;
    decoder_rnn_blob.dim2 = 1;

    // result shape: (seq_len * batch) * target_voc_size
    presoftmax_blob.dim0 = decoder_rnn_blob.dim0;
    presoftmax_blob.dim1 = target_voc_size;
    fc_compute.forward(&decoder_rnn_blob, &presoftmax_blob);

    // shape: (seq_len * batch) * target_voc_size
    softmax_result_blob.dim0 = presoftmax_blob.dim0;
    softmax_result_blob.dim1 = presoftmax_blob.dim1;
    softmax.forward(&presoftmax_blob, &softmax_result_blob);

    // shape: (seq_len * batch) * 1
    loss_blob.dim0 = softmax_result_blob.dim0;
    loss_blob.dim1 = 1;
    loss_compute.forward(&softmax_result_blob, decoder_target, &loss_blob);

    float total_loss = 0.0;
    int total_count = 0;

    loss_blob.copy_data_to_host();
    const float* loss_values = loss_blob.host_data;
    for (int i = 0; i < loss_blob.size(); ++i) {
        if (fabs(loss_values[i]) > 1e-12) {
            ++total_count;
            total_loss += loss_values[i];
        }
    }

#if 0
    fprintf(stderr, "encoder_input\n");
    encoder_input->copy_data_to_host();
    display_matrix(encoder_input->host_data, encoder_input->dim0, encoder_input->dim1);

    fprintf(stderr, "decoder_input\n");
    decoder_input->copy_data_to_host();
    display_matrix(decoder_input->host_data, decoder_input->dim0, decoder_input->dim1);

    fprintf(stderr, "decoder_target\n");
    decoder_target->copy_data_to_host();
    display_matrix(decoder_target->host_data, decoder_target->dim0, decoder_target->dim1);

    fprintf(stderr, "loss_blob\n");
    loss_blob.copy_data_to_host();
    display_matrix(loss_blob.host_data, loss_blob.dim0, loss_blob.dim1);

#endif
    float avg_loss = 0.0;
    loss_factor = 0.0;

    if (total_count > 0) {
        avg_loss = total_loss / total_count;
        loss_factor = 1.0 / total_count;
    }

    return avg_loss;
}

void Seq2SeqModel::backward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target) {
    loss_compute.backward(&softmax_result_blob, decoder_target, &loss_blob, loss_factor);
    softmax.backward(&presoftmax_blob, &softmax_result_blob);
    fc_compute.backward(&decoder_rnn_blob, &presoftmax_blob);

    decoder_rnn_blob.dim0 = decoder_embedding_blob.dim0;
    decoder_rnn_blob.dim1 = decoder_embedding_blob.dim1;
    decoder_rnn_blob.dim2 = hidden_size;

    // attention to this call
    decoder_rnn.backward(&decoder_embedding_blob,
            &encoder_rnn_blob,
            &decoder_rnn_blob);

    // attention to this call
    encoder_rnn.backward(&encoder_embedding_blob,
            &encoder_rnn_blob,
            NULL,
            NULL,
            NULL);

}
} // namespace seq2seq
