#include "data_reader.h"
#include "all_computes.h"
#include "model.h"
#include <gflags/gflags.h>
#include <sys/stat.h>

DEFINE_double(lr, 0.5, "learning rate");
DEFINE_int32(lr_decay_per_checkpoint, 0, "decay learning rate per this checkpoint (set it to 0 for auto decay)");
DEFINE_double(lr_decay, 0.999, "lr decay");
DEFINE_int32(max_iter, 5000, "max iter to train");
DEFINE_int32(batch_size, 32, "mini batch size");
DEFINE_int32(emb_size, 512, "embedding size");
DEFINE_int32(hidden_size, 1024, "hidden size");
DEFINE_int32(save_per_iter, 1000, "save model per this iters");
DEFINE_string(train_data_dir, "", "which folder to load training data");
DEFINE_string(save_model_dir, "", "which folder to save trained model");
DEFINE_string(load_model_dir, "", "which folder to load model (leave it empty for cold start)");
DEFINE_int32(checkpoint_per_iter, 200, "How many iters per checkpoint.");
DEFINE_double(max_gradient_norm, 5.0, "Clip gradients to this norm.");
DEFINE_int32(device, 0, "which device to use");
DEFINE_int32(max_source_len, 50, "max source language length for training");
DEFINE_int32(max_target_len, 50, "max target language length for training");

namespace seq2seq {
// TODO: using sampled softmax to reduce gpu memory cosuming on softmax compute
void run_train() {
//===========prepare data============
    DataReader reader;
    std::string source_vocab = FLAGS_train_data_dir + "/source.vocab";
    reader.load_source_dict(source_vocab.c_str());
    std::string target_vocab = FLAGS_train_data_dir + "/target.vocab";
    reader.load_target_dict(target_vocab.c_str());

    fprintf(stderr, "source dict size : %d, target dict size: %d\n",
            reader.source_dict_size(),
            reader.target_dict_size());

    int max_encoder_len = FLAGS_max_source_len;
    // target will addes eos and go_id
    int max_decoder_len = FLAGS_max_target_len + 2;

    std::string source_file_str = FLAGS_train_data_dir + "/source";
    std::string target_file_str = FLAGS_train_data_dir + "/target";
    reader.load_all_data(
            source_file_str.c_str(),
            target_file_str.c_str(),
            FLAGS_batch_size,
            FLAGS_max_source_len,
            FLAGS_max_target_len);

//================initialize model================
    Seq2SeqModel model;
    model.source_voc_size = reader.source_dict_size();
    model.target_voc_size = reader.target_dict_size();
    model.lr = FLAGS_lr;
    model.batch_size = FLAGS_batch_size;
    model.emb_size = FLAGS_emb_size;
    model.hidden_size = FLAGS_hidden_size;

    model.init(max_encoder_len, max_decoder_len);
    fprintf(stderr, "init ended\n");

    if (FLAGS_load_model_dir.size() > 0) {
        model.load_model(FLAGS_load_model_dir.c_str());
    }

    Blob encoder_input;
    Blob decoder_input;
    Blob decoder_target;

    encoder_input.dim0 = max_encoder_len;
    encoder_input.dim1 = FLAGS_batch_size;
    decoder_input.dim0 = max_decoder_len;
    decoder_input.dim1 = FLAGS_batch_size;
    decoder_target.dim0 = max_decoder_len;
    decoder_target.dim1 = FLAGS_batch_size;

    encoder_input.malloc_all_data();
    decoder_input.malloc_all_data();
    decoder_target.malloc_all_data();

    std::vector<float> losses;
//====================start optimization===========
    int checkpoint = 0;
    float checkpoint_avg_loss = 0.0;
    for (int iter = 0; iter < FLAGS_max_iter; ++iter) {
        reader.get_batch(&encoder_input, &decoder_input, &decoder_target);
        float iter_loss =  model.forward(&encoder_input, &decoder_input, &decoder_target);
        model.backward(&encoder_input, &decoder_input, &decoder_target);
        model.clip_gradients(FLAGS_max_gradient_norm);
        model.optimize(&encoder_input, &decoder_input);
        checkpoint_avg_loss += iter_loss;

        if ((iter + 1) % FLAGS_checkpoint_per_iter == 0) {
            ++checkpoint;
            float loss = checkpoint_avg_loss / FLAGS_checkpoint_per_iter;
            checkpoint_avg_loss = 0.0;

            float perplexity = exp(loss);
            fprintf(stderr, "iter = %d, lr = %6f, loss = %6f, perplexity = %6f\n",
                    iter + 1,
                    model.lr,
                    loss,
                    perplexity);

            losses.push_back(loss);

            if (FLAGS_lr_decay_per_checkpoint == 0) {
                float loss_decay = losses.size() > 10 && loss > losses[losses.size() - 4] ?
                        FLAGS_lr_decay : 1.0;
                model.lr *= loss_decay;
            } else if (checkpoint % FLAGS_lr_decay_per_checkpoint == 0) {
                model.lr *=  FLAGS_lr_decay;
            }
        }

        if ((iter + 1) % FLAGS_save_per_iter == 0) {
            std::string to_save_dir = FLAGS_save_model_dir + "/iter" + std::to_string(iter + 1);
            mkdir(to_save_dir.c_str(), 0777);
            fprintf(stderr, "saving model to %s\n", to_save_dir.c_str());
            model.save_model(to_save_dir.c_str());
        }
    }
    std::string to_save_dir = FLAGS_save_model_dir + "/final";
    mkdir(to_save_dir.c_str(), 0777);
    fprintf(stderr, "saving model to %s\n", to_save_dir.c_str());
    model.save_model(to_save_dir.c_str());
}
} // namespace seq2seq

int main(int argc, char** argv) {
//  srand(time(NULL));
    if (argc < 2) {
        fprintf(stderr, "use --help for details\n");
        exit(-1);
    }
    google::SetUsageMessage("seq2seq training");
    google::SetVersionString("1.0.0");
    google::ParseCommandLineFlags(&argc, &argv, true);
    cudaErrCheck(cudaSetDevice(FLAGS_device));

    if (FLAGS_train_data_dir.size() == 0
            || FLAGS_save_model_dir.size() == 0) {
        fprintf(stderr, "you must set train_data_dir and save_model_dir\n"
                "use --help for details\n");
        exit(-1);
    }

    struct stat sb;
    if (stat(FLAGS_save_model_dir.c_str(), &sb) != 0
            || !S_ISDIR(sb.st_mode)) {
        fprintf(stderr, "%s is not a valid path\n", FLAGS_save_model_dir.c_str());
        exit(-1);
    }

    seq2seq::run_train();

    google::ShutDownCommandLineFlags();
    return 0;
}
