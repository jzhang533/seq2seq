#ifndef SEQ2SEQ_INCLUDE_ALL_COMPUTES_H
#define SEQ2SEQ_INCLUDE_ALL_COMPUTES_H

#include "common.h"
#include "cudnn_util.h"

/*
 * breif: compute layers implemented by using cuDNN/cuBlas
 */

namespace seq2seq {

// this can be protected by a shared_ptr
class GpuMemPtr {
public:
    explicit GpuMemPtr() : _data(NULL) {
    }
    explicit GpuMemPtr(size_t size_in_bytes) {
        cudaErrCheck(cudaMalloc((void**)&_data, size_in_bytes));
    }
    ~GpuMemPtr() {
        if (_data != NULL) {
            cudaFree(_data);
        }
    }
    void* get() {
        return _data;
    }
private:
    void* _data;
    DISALLOW_COPY_AND_ASSIGN(GpuMemPtr);
};

// for simplicity, I use struct here
struct Blob {
    explicit Blob() : dim0(1), dim1(1), dim2(1) {}
    int dim0;
    int dim1;
    int dim2;
    float* host_data;
    float* host_diff;
    float* device_data;
    float* device_diff;

    inline int size() {
        return dim0 * dim1 * dim2;
    }

    inline void malloc_all_data() {
        // fprintf(stderr, "malloc size %d\n", size());
        host_data = (float*)malloc(size() * sizeof(float));
        assert(host_data != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_data, size() * sizeof(float)));
        cudaErrCheck(cudaMemset(device_data, 0.0, size() * sizeof(float)));

        host_diff = (float*)malloc(size() * sizeof(float));
        assert(host_diff != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_diff, size() * sizeof(float)));
        cudaErrCheck(cudaMemset(device_diff, 0.0, size() * sizeof(float)));
    }

    inline void copy_data_to_device() {
//        fprintf(stderr, "(%d, %d, %d) %d\n", dim0, dim1, dim2, size());
        cudaErrCheck(cudaMemcpy(device_data,
                host_data,
                size() * sizeof(float),
                cudaMemcpyHostToDevice));
    }

    inline void copy_data_to_host() {
        cudaErrCheck(cudaMemcpy(host_data,
                device_data,
                size() * sizeof(float),
                cudaMemcpyDeviceToHost));
    }

    inline void copy_diff_to_device() {
        cudaErrCheck(cudaMemcpy(device_diff,
                host_diff,
                size() * sizeof(float),
                cudaMemcpyHostToDevice));
    }

    inline void copy_diff_to_host() {
        cudaErrCheck(cudaMemcpy(host_diff,
                device_diff,
                size() * sizeof(float),
                cudaMemcpyDeviceToHost));
    }

    // saving matrix (ignore dim3) into a text file
    inline void savetxt(const char* filename) {
        this->copy_data_to_host();
        const float* data = this->host_data;
        FILE* fp = fopen(filename, "w");
        int row = dim0;
        int col = dim1 * dim2;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                fprintf(fp,
                        "%+.8f%s",
                        data[i * col + j],
                        j == col - 1 ? "" : " ");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    // loading matrix (ignore dim3) into a text file
    inline void loadtxt(const char* filename) {
        fprintf(stderr, "loading %s\n", filename);
        float* data = this->host_data;
        std::ifstream infile(filename);
        assert(infile.good());
        std::string line;
        std::vector<std::string> strs;
        int row = dim0;
        int col = dim1 * dim2;

        for (int i = 0; i < row; ++i) {
            getline(infile, line);
            split(line, strs);
            for (int j = 0; j < col; ++j) {
                data[i * col + j] = atof(strs[j].c_str());
//                fprintf(stderr, "===%.8f===\n", data[i*col +j]);
            }
        }
        this->copy_data_to_device();
    }

    // for debug purpose
    inline void display_data(const char* info = NULL) {
        this->copy_data_to_host();
        const char* str = info == NULL ? "" : info;
        fprintf(stderr, "%s in shape (%d %d %d)\n", str, dim0, dim1, dim2);
        display_matrix(this->host_data, dim0, dim1, dim2);
    }
    // for debug purpose
    inline void display_diff(const char* info = NULL) {
        this->copy_diff_to_host();
        const char* str = info == NULL ? "" : info;
        fprintf(stderr, "%s in shape (%d %d %d)\n", str, dim0, dim1, dim2);
        display_matrix(this->host_diff, dim0, dim1, dim2);
    }
};

class EmbCompute {
public:
    void init(int voc_size, int emb_size);
    void forward(Blob* input, Blob* output);
    void backward(Blob* input, Blob* output);

    void forward_for_rnn(Blob* input, Blob* output);
    void backward_for_rnn(Blob* input, Blob* output);
    inline void set_lr(float lr) {
        _lr = lr;
    }

    Blob* get_w() {
        return &_w;
    }
private:
    int _voc_size;
    int _emb_size;
    Blob _w;
    float _lr;
};

// FC
class FCCompute {
public:
    void init(int num_in, int num_out);
    void forward(Blob* input, Blob* output);
    void backward(Blob* input, Blob* output);
    Blob* get_w() {
        return &_w;
    }
    Blob* get_b() {
        return &_b;
    }
private:
    int _num_out;
    int _num_in;
    Blob _w;
    Blob _b;
    Blob _bias_multiplier;
    // this is used to create bias multiplier
    // an all one matrix for broadcasting
    const int max_allowd_batch = 40960;
};

class SoftmaxCompute {
public:
    /* use CUDNN_SOFTMAX_ACCURATE for inference
     * use CUDNN_SOFTMAX_LOG for training (together with other loss parts)
     */
    void init(cudnnSoftmaxAlgorithm_t alg = CUDNN_SOFTMAX_ACCURATE);
    void forward(Blob* input, Blob* output);
    void backward(Blob* input, Blob* output);
private:
    cudnnTensorDescriptor_t _input_desc;
    cudnnTensorDescriptor_t _output_desc;
    cudnnSoftmaxAlgorithm_t _alg;
};

// brief: NegativeLossCompute assumes the input is the result
// of SoftmaxCompute with CUDNN_SOFTMAX_LOG option
// a pad symbol id is required to set all of the diff to zero
// if that target is a pad symbol
// i.e.: only implements "negative" in negative log loss
//
class NegativeLossCompute {
public:
    inline void init(int pad_id = 0) {
        _pad_id = pad_id;
    }

    // input shape:  num_labels * batch
    // labels shape: batch * 1
    // output shape: batch * 1 (loss values)
    void forward(Blob* input, Blob* labels, Blob* output);
    void backward(Blob* input, Blob* labels, Blob* output, float loss_factor);
private:
    int _pad_id;
};

class ActivationCompute {
public:
    /*
     * possible mode:
     *  CUDNN_ACTIVATION_SIGMOID      = 0,
     *  CUDNN_ACTIVATION_RELU         = 1,
     *  CUDNN_ACTIVATION_TANH         = 2,
     *  CUDNN_ACTIVATION_CLIPPED_RELU = 3
     * TODO: when specifiying CLIPPED_RELU, set threshold
     */
    void init(cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID);
    void forward(Blob* input, Blob* output);
    void backward(Blob* input, Blob* output);
private:
    cudnnTensorDescriptor_t _input_desc;
    cudnnTensorDescriptor_t _output_desc;
    cudnnActivationMode_t _mode;
    cudnnActivationDescriptor_t  _activ_desc;
};

class RNNCompute {
public:
    void init(int batch_size,
            int hidden_size,
            int input_size,
            bool is_training = true,
            int num_layers = 1,
            bool bidirectional = false,
            cudnnRNNMode_t mode = CUDNN_GRU,
            float dropout = 0.0);

    /*
     * @brief forward pass of recurrent compute
     *
     * @param [in] input given x in shape: seq_length * batch * input_size
     * @param [out] output generate y in shape: seq_length * batch * hidden_size
     * @param [in] initial_hidden: in shape batch * input_size
     * @param [in] initial_cell: in shape batch * input_size
     * @param [out] final_hidden: in shape batch * hidden_size
     * @param [out] final_cell: in shape batch * hidden_size
     *
     * @return void
     * @retval none
     *
     */
    void forward(Blob* input,
            Blob* output,
            Blob* initial_hidden = NULL,
            Blob* initial_cell = NULL,
            Blob* final_hidden = NULL,
            Blob* final_cell = NULL);

    void backward(Blob* input,
            Blob* output,
            Blob* initial_hidden = NULL,
            Blob* initial_cell = NULL,
            Blob* final_hidden = NULL,
            Blob* final_cell = NULL);

    inline Blob* get_param_blob() {
        return &_param_blob;
    }

    inline std::vector<std::vector<Blob> >& get_matrix_blobs() {
        return _matrix_blobs;
    }
    inline std::vector<std::vector<Blob> >& get_bias_blobs() {
        return _bias_blobs;
    }

    inline float* get_dw() {
        return static_cast<float*>(_dw->get());
    }

    inline size_t weights_size() {
        return _weights_size;
    }

private:
    int _batch_size;
    int _hidden_size;
    int _input_size;
    bool _is_training;
    int _num_layers;
    bool _bidirectional;
    cudnnRNNMode_t _mode;
    float _dropout;

    // we will need this value to prepare descriptors for each time step
    static const int max_allowd_seq_length = 64;

    size_t _weights_size;
    cudnnFilterDescriptor_t _w_desc;        // all parameters descriptor
    cudnnFilterDescriptor_t _dw_desc;       // gradient of all parameters tensor descriptor
    std::shared_ptr<GpuMemPtr> _w;          // memory for all parameters
    std::shared_ptr<GpuMemPtr> _dw;         // memory for gradient of all parameters

    cudnnDropoutDescriptor_t _dropout_desc;
    std::shared_ptr<GpuMemPtr> _states; // memory for dropout internal

    size_t _work_size;
    size_t _reserve_size;
    std::shared_ptr<GpuMemPtr> _work_space; // memory for dropout internal
    std::shared_ptr<GpuMemPtr> _reserve_space; // memory for dropout internal

    cudnnRNNDescriptor_t _rnn_desc;

    cudnnTensorDescriptor_t _x_desc[max_allowd_seq_length];    // input tensor descriptors
    cudnnTensorDescriptor_t _y_desc[max_allowd_seq_length];    // hidden tensor descriptors
    cudnnTensorDescriptor_t _dx_desc[max_allowd_seq_length];   // gradient of input tensor descriptors
    cudnnTensorDescriptor_t _dy_desc[max_allowd_seq_length];   // gradient of hidden tensor descriptors

    cudnnTensorDescriptor_t _hx_desc;             // initial hidden tensor descriptor
    cudnnTensorDescriptor_t _cx_desc;             // initial cell tensor descriptor
    cudnnTensorDescriptor_t _hy_desc;             // final hidden tensor descriptor
    cudnnTensorDescriptor_t _cy_desc;             // final cell tensor descriptor

    cudnnTensorDescriptor_t _dhx_desc;             // gradient of initial hidden tensor descriptor
    cudnnTensorDescriptor_t _dcx_desc;             // gradient of initial cell tensor descriptor
    cudnnTensorDescriptor_t _dhy_desc;             // gradient of final hidden tensor descriptor
    cudnnTensorDescriptor_t _dcy_desc;             // gradient of final cell tensor descriptor
private:
    void initialize_params();
    // N.B. I will not use malloc_data of blob to manage the memory of param
    // instead, using codes in initialize_params to create/refer params
    Blob _param_blob;

    std::vector<std::vector<Blob> > _matrix_blobs;
    std::vector<std::vector<Blob> > _bias_blobs;
};
} // namespace seq2seq

#endif
