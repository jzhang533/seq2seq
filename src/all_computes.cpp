#include "all_computes.h"

namespace seq2seq {
void FCCompute::init(int num_in, int num_out) {
    _num_out = num_out;
    _num_in = num_in;

    // prepare weights
    _w.dim0 = num_in;
    _w.dim1 = num_out;
    _w.malloc_all_data();
    xavier_fill(_w.host_data, _w.size(), num_in, num_out);
//   constant_fill(_w.host_data, _w.size(), 1.0f);
    _w.copy_data_to_device();

    // prepare bias
    _b.dim0 = 1;
    _b.dim1 = num_out;
    _b.malloc_all_data();
    constant_fill(_b.host_data, num_out, 0.0f);
    _b.copy_data_to_device();

    // prepare bias_multiplier
    _bias_multiplier.dim0 = max_allowd_batch;
    _bias_multiplier.dim1 = 1;
    _bias_multiplier.malloc_all_data();
    constant_fill(_bias_multiplier.host_data, max_allowd_batch, 1.0f);
    _bias_multiplier.copy_data_to_device();
}

void FCCompute::forward(Blob* input, Blob* output) {
    // input dim0 * dim1 = batch size * num_input
    // weights dim0 * dim1 = num_input * num_out
    int batch = input->dim0;

    /*
    if (batch > max_allowd_batch) {
        fprintf(stderr, "fatal batch size exceeds %d\n", max_allowd_batch);
        exit(-1);
    }
    */

    gpu_gemm(CblasNoTrans,
            CblasNoTrans,
            batch,
            _num_out,
            _num_in,
            1.0f,
            input->device_data,
            _w.device_data,
            0.0f,
            output->device_data);

    // add bias
    gpu_gemm(CblasNoTrans,
            CblasNoTrans,
            batch,
            _num_out,
            1,
            1.0f,
            _bias_multiplier.device_data,
            _b.device_data,
            1.0f,
            output->device_data);
}

void FCCompute::backward(Blob* input, Blob* output) {
    int batch = input->dim0;
    // grads wrt w
    gpu_gemm(CblasTrans,
            CblasNoTrans,
            _num_in,
            _num_out,
            batch,
            1.0f,
            input->device_data,
            output->device_diff,
            0.0f,
            _w.device_diff);

    // grads wrt b
    gpu_gemv(CblasTrans,
            batch,
            _num_out,
            1.0f,
            output->device_diff,
            _bias_multiplier.device_data,
            0.0f,
            _b.device_diff);

    // grads wrt input
    gpu_gemm(CblasNoTrans,
            CblasTrans,
            batch,
            _num_in,
            _num_out,
            1.0f,
            output->device_diff,
            _w.device_data,
            0.0f,
            input->device_diff);
}

void EmbCompute::init(int voc_size, int emb_size) {
    _voc_size = voc_size;
    _emb_size = emb_size;

    // prepare weights
    _w.dim0 = voc_size;
    _w.dim1 = emb_size;
    _w.malloc_all_data();
    xavier_fill(_w.host_data, _w.size(), voc_size, emb_size);
    _w.copy_data_to_device();
}

void EmbCompute::forward(Blob* input, Blob* output) {
    // input: dim0 * dim1 = batch size * seq_length
    // weights: dim0 * dim1 = voc_size * emb_size
    // output: batch_size * seq_length * emb_size
    int batch = input->dim0;

    emb_ff(_w.device_data,
            input->device_data,
            output->device_data,
            batch,
            input->dim1,
            _emb_size);
}
void EmbCompute::backward(Blob* input, Blob* output) {
    // TODO: get real learning rate,
    // TODO: redesign to enable multi device training
    float mlr = -_lr;
    int batch = input->dim0;
    emb_bp(_w.device_data,
            input->device_data,
            output->device_diff,
            batch,
            input->dim1,
            _emb_size,
            mlr);
}

void EmbCompute::forward_for_rnn(Blob* input, Blob* output) {
    // input: dim0 * dim1 =  seq_length * batch_size
    // weights: dim0 * dim1 = voc_size * emb_size
    // output: seq_length * batch_size * emb_size
    int batch = input->dim0;

    emb_ff_for_rnn(_w.device_data,
            input->device_data,
            output->device_data,
            batch,
            input->dim1,
            _emb_size);
}

void EmbCompute::backward_for_rnn(Blob* input, Blob* output) {
    // TODO: redesign to enable multi device training
    float mlr = -_lr;
    int batch = input->dim0;
/*    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "mlr %f\n", mlr);
    fprintf(stderr, "input\n");
    input->display_data();
    fprintf(stderr, "output diff\n");
    output->display_diff();
    fprintf(stderr, "original w\n");
    _w.display_data();
    */

    emb_bp_for_rnn(_w.device_data,
            input->device_data,
            output->device_diff,
            batch,
            input->dim1,
            _emb_size,
            mlr);

//    fprintf(stderr, "now w\n");
//    _w.display_data();

//   fprintf(stderr, "===========================================\n");
}

void SoftmaxCompute::init(cudnnSoftmaxAlgorithm_t alg /* = CUDNN_SOFTMAX_ACCURATE */) {
    _alg = alg;
    cudnn::createTensor4dDesc<float>(&_input_desc);
    cudnn::createTensor4dDesc<float>(&_output_desc);
}

void SoftmaxCompute::forward(Blob* input, Blob* output) {
    // input dim0 * dim1 = batch size * num_labels
    int batch = input->dim0;
    int num_labels = input->dim1;
//    fprintf(stderr, "batch:%d num_labels:%d\n", batch, num_labels);

    int N = batch;
    int K = num_labels;
    int H = 1;
    int W = 1;
    cudnn::setTensor4dDesc<float>(&_input_desc, N, K, H, W);
    cudnn::setTensor4dDesc<float>(&_output_desc, N, K, H, W);

    cudnnErrCheck(cudnnSoftmaxForward(
            GlobalAssets::instance()->cudnnHandle(),
            _alg,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            cudnn::dataType<float>::one,
            _input_desc,
            input->device_data,
            cudnn::dataType<float>::zero,
            _output_desc,
            output->device_data));
}

void SoftmaxCompute::backward(Blob* input, Blob* output) {
    cudnnErrCheck(cudnnSoftmaxBackward(
            GlobalAssets::instance()->cudnnHandle(),
            _alg,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            cudnn::dataType<float>::one,
            _output_desc,
            output->device_data,
            _output_desc,
            output->device_diff,
            cudnn::dataType<float>::zero,
            _input_desc,
            input->device_diff));
}

void NegativeLossCompute::forward(Blob* input, Blob* labels, Blob* output) {
    int batch = input->dim0;
    int num_labels = input->dim1;
    negative_loss_ff(
            input->device_data,
            labels->device_data,
            output->device_data,
            batch,
            num_labels,
            _pad_id);
}

// output is the loss values, dont need for now
void NegativeLossCompute::backward(Blob* input, Blob* labels, Blob* output, float loss_factor) {
    int batch = input->dim0;
    int num_labels = input->dim1;
    negative_loss_bp(
            input->device_data,
            labels->device_data,
            input->device_diff,
            batch,
            num_labels,
            loss_factor,
            _pad_id);
}

void ActivationCompute::init(cudnnActivationMode_t mode /* = CUDNN_ACTIVATION_SIGMOID */) {
    _mode = mode;
    cudnnErrCheck(cudnnCreateActivationDescriptor(&_activ_desc));
    cudnnErrCheck(cudnnSetActivationDescriptor(_activ_desc,
            _mode,
            CUDNN_PROPAGATE_NAN,
            0.0));

    cudnn::createTensor4dDesc<float>(&_input_desc);
    cudnn::createTensor4dDesc<float>(&_output_desc);
}

void ActivationCompute::forward(Blob* input, Blob* output) {
    // input dim0 * dim1 = batch size * num
    int batch = input->dim0;
    int num = input->dim1;

    int N = batch;
    int K = num;
    int H = 1;
    int W = 1;
    cudnn::setTensor4dDesc<float>(&_input_desc, N, K, H, W);
    cudnn::setTensor4dDesc<float>(&_output_desc, N, K, H, W);

    cudnnErrCheck(cudnnActivationForward(
            GlobalAssets::instance()->cudnnHandle(),
            _activ_desc,
            cudnn::dataType<float>::one,
            _input_desc,
            input->device_data,
            cudnn::dataType<float>::zero,
            _output_desc,
            output->device_data));
}

void ActivationCompute::backward(Blob* input, Blob* output) {
    cudnnErrCheck(cudnnActivationBackward(
            GlobalAssets::instance()->cudnnHandle(),
            _activ_desc,
            cudnn::dataType<float>::one,
            _output_desc,
            output->device_data,
            _output_desc,
            output->device_diff,
            _input_desc,
            input->device_data,
            cudnn::dataType<float>::zero,
            _input_desc,
            input->device_diff));
}

void RNNCompute::init(int batch_size,
        int hidden_size,
        int input_size,
        bool is_training /* = true */,
        int num_layers /* = 1 */,
        bool bidirectional /* = false */,
        cudnnRNNMode_t mode /* = CUDNN_GRU */,
        float dropout /* = 0.0 */) {
    _batch_size = batch_size;
    _hidden_size = hidden_size;
    _input_size = input_size;
    _is_training = is_training;
    _num_layers = num_layers;
    _bidirectional = bidirectional;
    _mode = mode;
    _dropout = dropout;

   // -------------------------
   // Set up inputs and outputs
   // -------------------------

   int dimA[3];
   int strideA[3];

   // In this example dimA[1] is constant across the whole sequence
   // This isn't required, all that is required is that it does not increase.
   for (unsigned int i = 0; i < max_allowd_seq_length; ++i) {
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_x_desc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_y_desc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_dx_desc[i]));
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_dy_desc[i]));

      dimA[0] = _batch_size;
      dimA[1] = _input_size;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(_x_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(_dx_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

      dimA[0] = _batch_size;
      dimA[1] = _bidirectional ? _hidden_size * 2 : _hidden_size;
      dimA[2] = 1;

      strideA[0] = dimA[2] * dimA[1];
      strideA[1] = dimA[2];
      strideA[2] = 1;

      cudnnErrCheck(cudnnSetTensorNdDescriptor(_y_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
      cudnnErrCheck(cudnnSetTensorNdDescriptor(_dy_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
   }


   dimA[0] = _num_layers * (_bidirectional ? 2 : 1);
   dimA[1] = _batch_size;
   dimA[2] = _hidden_size;

   strideA[0] = dimA[2] * dimA[1];
   strideA[1] = dimA[2];
   strideA[2] = 1;

   cudnnErrCheck(cudnnCreateTensorDescriptor(&_hx_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_cx_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_hy_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_cy_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_dhx_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_dcx_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_dhy_desc));
   cudnnErrCheck(cudnnCreateTensorDescriptor(&_dcy_desc));


   cudnnErrCheck(cudnnSetTensorNdDescriptor(_hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_dhx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_dcx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_dhy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
   cudnnErrCheck(cudnnSetTensorNdDescriptor(_dcy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


   // -------------------------
   // Set up the dropout descriptor (needed for the RNN descriptor)
   // -------------------------
   static const unsigned long long seed = 1337ull; // Pick a seed.

   cudnnErrCheck(cudnnCreateDropoutDescriptor(&_dropout_desc));

   // How much memory does dropout need for states?
   // These states are used to generate random numbers internally
   // and should not be freed until the RNN descriptor is no longer used
   size_t state_size;
   cudnnErrCheck(cudnnDropoutGetStatesSize(
            GlobalAssets::instance()->cudnnHandle(),
            &state_size));

   _states.reset(new GpuMemPtr(state_size));

   cudnnErrCheck(cudnnSetDropoutDescriptor(_dropout_desc,
            GlobalAssets::instance()->cudnnHandle(),
            _dropout,
            _states->get(),
            state_size,
            seed));

   // -------------------------
   // Set up the RNN descriptor
   // -------------------------
   cudnnErrCheck(cudnnCreateRNNDescriptor(&_rnn_desc));
   cudnnErrCheck(cudnnSetRNNDescriptor(_rnn_desc,
            _hidden_size,
            _num_layers,
            _dropout_desc,
            CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
            _bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            _mode,
            CUDNN_DATA_FLOAT));

   // -------------------------
   // Set up parameters
   // -------------------------
   // This needs to be done after the rnn descriptor is set as otherwise
   // we don't know how many parameters we have to allocate
   cudnnErrCheck(cudnnCreateFilterDescriptor(&_w_desc));
   cudnnErrCheck(cudnnCreateFilterDescriptor(&_dw_desc));

   cudnnErrCheck(cudnnGetRNNParamsSize(
            GlobalAssets::instance()->cudnnHandle(),
            _rnn_desc,
            _x_desc[0],
            &_weights_size,
            CUDNN_DATA_FLOAT));

//   fprintf(stderr, "rnn parameter size = %ld float numbers\n", _weights_size / sizeof(float));

   int dimW[3];
   dimW[0] =  _weights_size / sizeof(float);
   dimW[1] = 1;
   dimW[2] = 1;

   cudnnErrCheck(cudnnSetFilterNdDescriptor(_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
   cudnnErrCheck(cudnnSetFilterNdDescriptor(_dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

   _w.reset(new GpuMemPtr(_weights_size));
   _dw.reset(new GpuMemPtr(_weights_size));

   // -------------------------
   // Set up work space and reserved memory
   // -------------------------
   // Need for every pass
   cudnnErrCheck(cudnnGetRNNWorkspaceSize(
            GlobalAssets::instance()->cudnnHandle(),
            _rnn_desc,
            max_allowd_seq_length,
            _x_desc,
            &_work_size));

   // Only needed in training, shouldn't be touched between passes.
   cudnnErrCheck(cudnnGetRNNTrainingReserveSize(
            GlobalAssets::instance()->cudnnHandle(),
            _rnn_desc,
            max_allowd_seq_length,
            _x_desc,
            &_reserve_size));

   _work_space.reset(new GpuMemPtr(_work_size));
   _reserve_space.reset(new GpuMemPtr(_reserve_size));

   // Finally, initialize params
   this->initialize_params();
}

void RNNCompute::initialize_params() {
    // use queryed matrixes and biases
    int num_linear_layers = 0;
    if (_mode == CUDNN_RNN_RELU || _mode == CUDNN_RNN_TANH) {
        num_linear_layers = 2;
    } else if (_mode == CUDNN_LSTM) {
        num_linear_layers = 8;
    } else if (_mode == CUDNN_GRU) {
        num_linear_layers = 6;
    }

    int total_layers= _num_layers * (_bidirectional ? 2 : 1);
    _matrix_blobs.resize(total_layers);
    _bias_blobs.resize(total_layers);

    for (int layer = 0; layer < total_layers; ++layer) {
//        fprintf(stderr, "layer:%d\n", layer);
        _matrix_blobs[layer].resize(num_linear_layers);
        _bias_blobs[layer].resize(num_linear_layers);

        for (int lin_layer_id = 0; lin_layer_id < num_linear_layers; ++lin_layer_id) {
//            fprintf(stderr, "lin_layer_id:%d\n", lin_layer_id);
            Blob& matrix = _matrix_blobs[layer][lin_layer_id];
            Blob& bias = _bias_blobs[layer][lin_layer_id];

            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            float *linLayerMat;

            cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(
                    GlobalAssets::instance()->cudnnHandle(),
                    _rnn_desc,
                    layer,
                    _x_desc[0],
                    _w_desc,
                    _w->get(),
                    lin_layer_id,
                    linLayerMatDesc,
                    (void**)&linLayerMat));

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];
            cudnnErrCheck(cudnnGetFilterNdDescriptor(
                    linLayerMatDesc,
                    3,
                    &dataType,
                    &format,
                    &nbDims,
                    filterDimA));

//            fprintf(stderr,"filterDimA: %d %d %d\n", filterDimA[0], filterDimA[1], filterDimA[2]);
            initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 0.1f);
            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

            matrix.device_data = static_cast<float*>(linLayerMat);
            //matrix.dim0 = filterDimA[0];
            matrix.dim0 = lin_layer_id < num_linear_layers / 2 ?  _input_size : _hidden_size;
            matrix.dim1 = _hidden_size;

            // get bias
            cudnnFilterDescriptor_t linLayerBiasDesc;
            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            float *linLayerBias;

            cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(
                    GlobalAssets::instance()->cudnnHandle(),
                    _rnn_desc,
                    layer,
                    _x_desc[0],
                    _w_desc,
                    _w->get(),
                    lin_layer_id,
                    linLayerBiasDesc,
                    (void**)&linLayerBias));

            cudnnErrCheck(cudnnGetFilterNdDescriptor(
                    linLayerBiasDesc,
                    3,
                    &dataType,
                    &format,
                    &nbDims,
                    filterDimA));

//            fprintf(stderr,"filterDimA: %d %d %d\n", filterDimA[0], filterDimA[1], filterDimA[2]);
            initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 0.1);

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));

            bias.device_data = static_cast<float*>(linLayerBias);
            bias.dim0 = _hidden_size;
        }
    }

    // fill matrix
    for (size_t i = 0; i < _matrix_blobs.size(); ++i) {
        for (size_t j = 0; j < _matrix_blobs[i].size(); ++j) {
            Blob& matrix = _matrix_blobs[i][j];

            matrix.host_data = (float*)malloc(matrix.size() * sizeof(float));
            assert(matrix.host_data != NULL);

            matrix.host_diff = (float*)malloc(matrix.size() * sizeof(float));
            assert(matrix.host_diff != NULL);

            xavier_fill(matrix.host_data, matrix.size(), matrix.dim0, matrix.dim1);
            matrix.copy_data_to_device();
        }
    }

    // fill bias
    for (size_t i = 0; i < _bias_blobs.size(); ++i) {
        for (size_t j = 0; j < _bias_blobs[i].size(); ++j) {
            Blob& bias = _bias_blobs[i][j];

            bias.host_data = (float*)malloc(bias.size() * sizeof(float));
            assert(bias.host_data != NULL);

            bias.host_diff = (float*)malloc(bias.size() * sizeof(float));
            assert(bias.host_diff != NULL);

            constant_fill(bias.host_data, bias.size(), 0.0f);
            bias.copy_data_to_device();
        }
    }

    // use a whole blob for all of the parameters
    _param_blob.dim0 = _weights_size / sizeof(float);
    _param_blob.dim1 = 1;
    _param_blob.dim2 = 1;

    // TODO: be aware of this usage, especially when adding codes to free these memory
    _param_blob.device_data = static_cast<float*>(_w->get());
    _param_blob.device_diff = static_cast<float*>(_dw->get());

    _param_blob.host_data = (float*)malloc(_param_blob.size() * sizeof(float));
    assert(_param_blob.host_data != NULL);
    _param_blob.host_diff = (float*)malloc(_param_blob.size() * sizeof(float));
    assert(_param_blob.host_diff != NULL);
}

void RNNCompute::forward(Blob* input,
        Blob* output,
        Blob* initial_hidden /* = NULL */,
        Blob* initial_cell /* = NULL */,
        Blob* final_hidden /* = NULL */,
        Blob* final_cell /* = NULL */) {

    int seq_length = input->dim0;
    int batch = input->dim1;
    int input_size = input->dim2;

    if (seq_length > max_allowd_seq_length) {
        fprintf(stderr, "length (%d) exceeds (%d)\n", seq_length, max_allowd_seq_length);
        exit(-1);
    }
    if (batch != _batch_size) {
        fprintf(stderr, "batch (%d) not same as stated in init (%d)\n",
                batch,
                _batch_size);
        exit(-1);
    }
    if (input_size != _input_size) {
        fprintf(stderr, "input_size (%d) not same as stated in init (%d)\n",
                input_size,
                _input_size);
        exit(-1);
    }

    if (_is_training) {
        cudnnErrCheck(cudnnRNNForwardTraining(
                GlobalAssets::instance()->cudnnHandle(),
                _rnn_desc,
                seq_length,
                _x_desc,
                input->device_data,
                _hx_desc,
                initial_hidden == NULL ? NULL : initial_hidden->device_data,
                _cx_desc,
                initial_cell == NULL ? NULL : initial_cell->device_data,
                _w_desc,
                _w->get(),
                _y_desc,
                output->device_data,
                _hy_desc,
                final_hidden == NULL ? NULL : final_hidden->device_data,
                _cy_desc,
                final_cell == NULL ? NULL : final_cell->device_data,
                _work_space->get(),
                _work_size,
                _reserve_space->get(),
                _reserve_size));
    } else {
        // TODO: call inferce
    }
}

void RNNCompute::backward(Blob* input,
        Blob* output,
        Blob* initial_hidden /* = NULL */,
        Blob* initial_cell /* = NULL */,
        Blob* final_hidden /* = NULL */,
        Blob* final_cell /* = NULL */) {

    int seq_length = input->dim0;
    int batch = input->dim1;
    int input_size = input->dim2;

    if (seq_length > max_allowd_seq_length) {
        fprintf(stderr, "length (%d) exceeds (%d)\n", seq_length, max_allowd_seq_length);
        exit(-1);
    }
    if (batch != _batch_size) {
        fprintf(stderr, "batch (%d) not same as stated in init (%d)\n",
                batch,
                _batch_size);
        exit(-1);
    }
    if (input_size != _input_size) {
        fprintf(stderr, "input_size (%d) not same as stated in init (%d)\n",
                input_size,
                _input_size);
        exit(-1);
    }

    // call backward
    cudnnErrCheck(cudnnRNNBackwardData(
            GlobalAssets::instance()->cudnnHandle(),
            _rnn_desc,
            seq_length,
            _y_desc,
            output->device_data,
            _dy_desc,
            output->device_diff,
            _dhy_desc,
            final_hidden == NULL ? NULL : final_hidden->device_diff,
            _dcy_desc,
            final_cell == NULL ? NULL : final_cell->device_diff,
            _w_desc,
            _w->get(),
            _hx_desc,
            initial_hidden == NULL ? NULL : initial_hidden->device_data,
            _cx_desc,
            initial_cell == NULL ? NULL : initial_cell->device_data,
            _dx_desc,
            input->device_diff,
            _dhx_desc,
            initial_hidden == NULL ? NULL : initial_hidden->device_diff,
            _dcx_desc,
            initial_cell == NULL ? NULL : initial_cell->device_diff,
            _work_space->get(),
            _work_size,
            _reserve_space->get(),
            _reserve_size));

   // cudnnRNNBackwardWeights adds to the data in dw.
   cudaErrCheck(cudaMemset(_dw->get(), 0, _weights_size));

   cudnnErrCheck(cudnnRNNBackwardWeights(
            GlobalAssets::instance()->cudnnHandle(),
            _rnn_desc,
            seq_length,
            _x_desc,
            input->device_data,
            _hx_desc,
            initial_hidden == NULL ? NULL : initial_hidden->device_data,
            _y_desc,
            output->device_data,
            _work_space->get(),
            _work_size,
            _dw_desc,
            _dw->get(),
            _reserve_space->get(),
            _reserve_size));
}
} // namespace seq2seq

