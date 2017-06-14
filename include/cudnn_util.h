#ifndef SEQ2SEQ_INCLUDE_CUDNN_UTIL_H
#define SEQ2SEQ_INCLUDE_CUDNN_UTIL_H
#include <cudnn.h>
#include "common.h"

namespace seq2seq {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};

template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
    cudnnErrCheck(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
  int n, int c, int h, int w,
  int stride_n, int stride_c, int stride_h, int stride_w) {
    cudnnErrCheck(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
            n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
  int n, int c, int h, int w) {
    const int stride_w = 1;
    const int stride_h = w * stride_w;
    const int stride_c = h * stride_h;
    const int stride_n = c * stride_c;
    setTensor4dDesc<Dtype>(desc, n, c, h, w,
            stride_n, stride_c, stride_h, stride_w);
}

}  // namespace cudnn

} // namespace seq2seq

#endif
