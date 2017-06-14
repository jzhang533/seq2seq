#include "data_reader.h"
#include "all_computes.h"

namespace seq2seq {

static const int max_iter = 1000;
static const int total_example = 10000;
static const int batch_size = 16;
static const int num_features = 128;
static const int num_labels = 4;

void print_loss(Blob& loss_blob) {
    float total_loss = 0.0;
    int total_count = 0;

    loss_blob.copy_data_to_host();
    const float* loss_values = loss_blob.host_data;
    for (int i = 0; i < loss_blob.size(); ++i) {
        if (fabs(loss_values[i]) > 1e-8) {
            ++total_count;
            total_loss += loss_values[i];
        }
    }

    assert(total_count > 0);
    float avg_loss = total_loss / total_count;
    float perplexity = exp(avg_loss);
    fprintf(stderr, "loss = %6f (%6f / %d), perplexity = %6f\n",
            avg_loss,
            total_loss,
            total_count,
            perplexity);
}

void test_negative_loss() {
    Blob all_examples;
    Blob all_labels;

    all_examples.dim0 = total_example;
    all_examples.dim1 = num_features;
    all_examples.malloc_all_data();

    all_labels.dim0 = total_example;
    all_labels.dim1 = 1;
    all_labels.malloc_all_data();

    for (int i = 0; i < all_examples.size(); ++i) {
        all_examples.host_data[i] = uniform_rand(0.0, 1.0);
    }
    for (int i = 0; i < all_labels.size(); ++i) {
        all_labels.host_data[i] = static_cast<float>(rand() % num_labels);
    }

    Blob input;
    input.dim0 = batch_size;
    input.dim1 = num_features;
    input.malloc_all_data();

    Blob labels;
    labels.dim0 = batch_size;
    labels.dim1 = 1;
    labels.malloc_all_data();

    Blob presoftmax;
    presoftmax.dim0 = batch_size;
    presoftmax.dim1 = num_labels;
    presoftmax.malloc_all_data();

    Blob softmaxresult;
    softmaxresult.dim0 = batch_size;
    softmaxresult.dim1 = num_labels;
    softmaxresult.malloc_all_data();

    Blob loss;
    loss.dim0 = batch_size;
    loss.dim1 = 1;
    loss.malloc_all_data();

    FCCompute fc;
    fc.init(num_features, num_labels);

    SoftmaxCompute softmax;
    softmax.init(CUDNN_SOFTMAX_LOG);

    NegativeLossCompute loss_compute;
    loss_compute.init(num_labels + 10);

    for (int iter = 0; iter < max_iter; ++iter) {
        // prepare data
        for (int i = 0; i < batch_size; ++i) {
            int idx = rand() % total_example;
            std::copy(
                    all_examples.host_data + idx * num_features,
                    all_examples.host_data + idx * num_features + num_features,
                    input.host_data + i * num_features);
            labels.host_data[i] = all_labels.host_data[idx];
        }
        input.copy_data_to_device();
        labels.copy_data_to_device();

        input.display_data("input data");
        fc.forward(&input, &presoftmax);
        presoftmax.display_data("presoftmax data");
        softmax.forward(&presoftmax, &softmaxresult);

        softmaxresult.display_data("softmaxresult data");
        labels.display_data("labels data");
        loss_compute.forward(&softmaxresult, &labels, &loss);
        loss.display_data("loss data");

        print_loss(loss);

        loss_compute.backward(&softmaxresult, &labels, &loss, 1.0 / batch_size);
        softmaxresult.display_diff("softmaxresult diff");

        softmax.backward(&presoftmax, &softmaxresult);
        presoftmax.display_diff("presoftmax diff");

        fc.backward(&input, &presoftmax);

        float mlr = -0.01;

        fprintf(stderr, "===================fc w data==============\n\n\n");
        fc.get_w()->display_data();
        fprintf(stderr, "===================fc w diff==============\n");
        fc.get_w()->display_diff();
        fprintf(stderr, "===================mlr %f==============\n", mlr);

        cublasErrCheck(cublasSaxpy(
                GlobalAssets::instance()->cublasHandle(),
                fc.get_w()->size(),
                &mlr,
                fc.get_w()->device_diff,
                1,
                fc.get_w()->device_data,
                1));
    }
}

} // namespace seq2seq

int main(int argc, char** argv) {
//  srand(time(NULL));
    seq2seq::test_negative_loss();
    return 0;
}
