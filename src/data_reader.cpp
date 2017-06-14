#include "data_reader.h"
#include <iostream>
#include <algorithm>
#include <fstream>

namespace seq2seq {
const int DataReader::PAD_ID = 0;
const int DataReader::GO_ID = 1;
const int DataReader::EOS_ID = 2;
const int DataReader::UNK_ID = 3;

// TODO: reduce risky in this constructor, e.g.: atoi, and split
TrainPair::TrainPair(const std::vector<int>& source_vec, const std::vector<int>& target_vec) {
    source_ids = new int[source_vec.size()];
    assert(source_ids != NULL);

    target_ids = new int[target_vec.size()];
    assert(target_ids != NULL);

    source_len = source_vec.size();
    target_len = target_vec.size();

    for (int i = 0; i < source_len; ++i) {
        source_ids[i] = source_vec[i];
    }

    for (int i = 0; i < target_len; ++i) {
        target_ids[i] = target_vec[i];
    }
}

void DataReader::load_all_data(const char* source_file,
        const char* target_file,
        const int batch_size /* = 32 */,
        const int max_source_len /* = 50 */,
        const int max_target_len /* = 50 */) {
    _batch_size = batch_size;
    _max_source_len = max_source_len;
    _max_target_len = max_target_len;

    std::ifstream source_f(source_file);
    std::ifstream target_f(target_file);

    if (!source_f.good()) {
        fprintf(stderr, "error opening file %s\n", source_file);
        exit(-1);
    }

    if (!target_f.good()) {
        fprintf(stderr, "error opening file %s\n", target_file);
        exit(-1);
    }

    std::string source_line;
    std::string target_line;

    size_t total_count = 0;
    size_t used_count = 0;

    size_t line_counter = 0;
    std::vector<int> source_ids;
    std::vector<int> target_ids;
    while (getline(source_f, source_line) && getline(target_f, target_line)) {
        ++line_counter;
        if (line_counter % 10000 == 0) {
            fprintf(stderr, "%ld lines loaded\n", line_counter);
        }
#if 0
        fprintf(stderr, "line %d, source:[%s], target[%s]\n",
                line_counter,
                source_line.c_str(),
                target_line.c_str());
#endif

        this->str_to_ids(source_line, source_ids, _source_dict_map);
        this->str_to_ids(target_line, target_ids, _target_dict_map);

        ++total_count;
        if ((int)source_ids.size() > _max_source_len || (int)target_ids.size() > _max_target_len) {
            fprintf(stderr, "skipped line %ld, (%ld, %ld) exceeds (%d, %d)\n",
                    line_counter,
                    source_ids.size(),
                    target_ids.size(),
                    _max_source_len,
                    _max_target_len);
            continue;
        }

        ++used_count;

        // insert a eos into target_ids
        target_ids.push_back(EOS_ID);

        _all_data.push_back(std::shared_ptr<TrainPair>(
                new TrainPair(source_ids, target_ids)));
    }

    fprintf(stderr, "%ld examples used in %ld\n", used_count, total_count);
    this->prepare_data_set();
}

bool DataReader::prefetch() {
    _prefetched_examples.clear();
    _prefetch_cursor = 0;

    if (_cursor >= _all_data.size()) {
        return false;
    }
    size_t total_fetch = _batch_size * prefetch_batch_count;
#if 0
    fprintf(stderr, "total: %ld total_fetch:%ld batch_size:%d prefetch_batch_count:%ld\n",
            _all_data.size(), total_fetch, _batch_size, prefetch_batch_count);
#endif

    // pick from all_data sequentially
    size_t k = 0;
    size_t sequential_pick_left = _all_data.size() - _cursor;
    for (; k < std::min(total_fetch, sequential_pick_left); ++k) {
        _prefetched_examples.push_back(_example_idx.get()[_cursor]);
        ++_cursor;
    }

    // pick randomly if no enough data left
    for (; k < total_fetch; ++k) {
        // randomly select from training examples
        size_t idx = rand() % _all_data.size();
        _prefetched_examples.push_back(idx);
    }

    // sort the prefetched examples
    std::sort(_prefetched_examples.begin(), _prefetched_examples.end(),
            [this](size_t a, size_t b) {
                if (this->_all_data[a]->source_len < this->_all_data[b]->source_len) {
                    return true;
                } else if (this->_all_data[a]->source_len == this->_all_data[b]->source_len) {
                    return this->_all_data[a]->target_len < this->_all_data[b]->target_len;
                } else {
                    return false;
                }
            });

    return true;
}

void DataReader::prepare_data_set() {
    // start from the first query of the dataset
    _cursor = 0;
    _prefetch_cursor = 0;

    size_t example_num = _all_data.size();
    fprintf(stderr, "total example_num: %ld, now building index\n", example_num);
    _example_idx.reset(new size_t[example_num]);
    // build index
    for (size_t i = 0; i < example_num; ++i) {
        _example_idx.get()[i] = i;
    }

    fprintf(stderr, "shuffling examples\n");
    std::random_shuffle(_example_idx.get(), _example_idx.get() + example_num);
    prefetch();
}

//
// return inputs in shape (
// encoder_input : encoder_size * batch,
// decoder_input : decoder_size * batch
// decoder_target : decoder_size * batch
//
void DataReader::get_batch(
        Blob* encoder_input,
        Blob* decoder_input,
        Blob* decoder_target) {

    size_t batch_end = _prefetch_cursor + _batch_size;
    int encoder_size = _all_data[_prefetched_examples[batch_end - 1]]->source_len;
    auto max_decoder_ele = std::max_element(
            _prefetched_examples.begin() + _prefetch_cursor,
            _prefetched_examples.begin() + _prefetch_cursor + _batch_size,
            [this](size_t const& a, size_t const& b) {
                return this->_all_data[a]->target_len < this->_all_data[b]->target_len;
            });

    int decoder_size = _all_data[*max_decoder_ele]->target_len + 1;

    float* encoder_input_data_t = encoder_input->host_data;
    float* decoder_input_data_t = decoder_input->host_data;
    float* decoder_target_data_t = decoder_target->host_data;
    for (int i = 0; i < _batch_size; ++i) {
        const TrainPair* train_pair = _all_data[_prefetched_examples[_prefetch_cursor + i]].get();

        int k = i;
        // pad encoder input
        for (; k < (encoder_size - train_pair->source_len) * _batch_size; k += _batch_size) {
            encoder_input_data_t[k] = static_cast<float>(PAD_ID);
        }
#if 0
        // reverse encoder input
        for (int j = train_pair->source_len - 1; j >=0; --j) {
            encoder_input_data_t[k] = static_cast<float>(train_pair->source_ids[j]);
            k += _batch_size;
        }
#else
        for (int j = 0; j < train_pair->source_len; ++j) {
            encoder_input_data_t[k] = static_cast<float>(train_pair->source_ids[j]);
            k += _batch_size;
        }
#endif

        // insert GO_ID into decoder begining
        k = i;
        decoder_input_data_t[k] = static_cast<float>(GO_ID);
        k += _batch_size;

        // copy decoder real inputs
        for (int j = 0; j < train_pair->target_len; ++j) {
            decoder_input_data_t[k] = static_cast<float>(train_pair->target_ids[j]);
            k += _batch_size;
        }
        // pad decoder input
        for (; k < decoder_size * _batch_size; k += _batch_size) {
            decoder_input_data_t[k] = static_cast<float>(PAD_ID);
        }

        // decoder_target is a shift of decoder_input
        k = i;
        for (; k < (decoder_size - 1) * _batch_size; k += _batch_size) {
            decoder_target_data_t[k] = decoder_input_data_t[k + _batch_size];
        }
        // use pad on last one
        decoder_target_data_t[k] = static_cast<float>(PAD_ID);
    }
    encoder_input->dim0 = encoder_size;
    encoder_input->dim1 = _batch_size;
    decoder_input->dim0 = decoder_size;
    decoder_input->dim1 = _batch_size;
    decoder_target->dim0 = decoder_size;
    decoder_target->dim1 = _batch_size;

#if 0
    fprintf(stderr, "copying data to device encoder: %ld, decoder:%ld, decoder_target:%ld \n",
            encoder_size, decoder_size, decoder_size);
#endif

    encoder_input->copy_data_to_device();
    decoder_input->copy_data_to_device();
    decoder_target->copy_data_to_device();


    _prefetch_cursor += _batch_size;
    if (_prefetch_cursor >= _prefetched_examples.size()) {
        if(!prefetch()) {
            size_t example_num = _all_data.size();
            fprintf(stderr, "shuffling examples\n");
            std::random_shuffle(_example_idx.get(), _example_idx.get() + example_num);
            _cursor = 0;
            _prefetch_cursor = 0;
            prefetch();
        }
    }
}

void DataReader::display_all_data() {
    for (const auto& item : _all_data) {
        fprintf(stderr, "source\n");
        display_matrix(item->source_ids, 1, item->source_len);
        fprintf(stderr, "target\n");
        display_matrix(item->target_ids, 1, item->target_len);
    }
}

void DataReader::load_source_dict(const char* source_vocab) {
    _source_vocab = source_vocab;
    this->load_dict(source_vocab, _source_dict_map, _rev_source_dict_vec);
}

void DataReader::load_target_dict(const char* target_vocab) {
    _target_vocab = target_vocab;
    this->load_dict(target_vocab, _target_dict_map, _rev_target_dict_vec);
}

void DataReader::load_dict(const char* vocab_file,
        std::unordered_map<std::string, int>& dict,
        std::vector<std::string>& rev_dict) {

    std::ifstream file(vocab_file);
    if (!file.good()) {
        fprintf(stderr, "error opening file %s\n", vocab_file);
        exit(-1);
    }

    std::string line;
    int index = 0;
    while (getline(file, line)) {
//        fprintf(stderr, "line:%d [%s]\n", index, line.c_str());
        if (dict.count(line) > 0) {
            fprintf(stderr, "duplicated entry:[%s] in file %s\n", line.c_str(), vocab_file);
            exit(-1);
        }
        dict[line] = index;
        ++index;
        rev_dict.push_back(line);
    }

    // check top 4 entries
    // TODO: make this check optional
    if (rev_dict.size() < 4
            || rev_dict[0] != "_PAD"
            || rev_dict[1] != "_GO"
            || rev_dict[2] != "_EOS"
            || rev_dict[3] != "_UNK") {
        fprintf(stderr, "top four words in dict should be: _PAD, _GO, _EOS, _UNK\n");
        exit(-1);
    }
}

void DataReader::str_to_ids(const std::string& str,
        std::vector<int>& result,
        const std::unordered_map<std::string, int>& dict) {
    std::vector<std::string> tokens;
    split(str, tokens);

    result.resize(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (dict.count(tokens[i]) > 0) {
            result[i] = dict.at(tokens[i]);
        } else {
            fprintf(stderr, "[%s] mapped to UNK in [%s]\n",
                    tokens[i].c_str(),
                    str.c_str());
            result[i] = UNK_ID;
            //exit(-1);
        }
    }
}

} // namespace seq2seq

