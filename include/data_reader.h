#ifndef SEQ2SEQ_INCLUDE_DATA_READER_H
#define SEQ2SEQ_INCLUDE_DATA_READER_H

#include "common.h"
#include "all_computes.h"
#include <string>
#include <map>
#include <unordered_map>

/*
 * breif: a helper class to load/prepare data for training
 *
 * unlike in tensorflow, where vocab is created in python,
 * here, I assume the voacbulary file has already been created
 * and the source/target training has alread been tokenized
 * (should be consistent with corresponding vocabularies)
 *
 */

namespace seq2seq {

// brief: a training example
struct TrainPair {
    TrainPair(): source_ids(NULL), target_ids(NULL), source_len(0), target_len(0) {
    }
    TrainPair(const std::vector<int>& source_vec, const std::vector<int>& target_vec);
    ~TrainPair() {
        if (source_ids != NULL) {
            delete []source_ids;
        }
        if (target_ids != NULL) {
            delete []target_ids;
        }
    }

    int* source_ids;
    int* target_ids;
    int source_len;
    int target_len;
};

// this can be protected by a shared_ptr
class DataReader {
public:
    static const int PAD_ID/* = 0 */;
    static const int GO_ID/* = 1 */;
    static const int EOS_ID/* = 2 */;
    static const int UNK_ID/* = 3*/;
public:
    explicit DataReader() {
    }
    ~DataReader() {
    }

    void load_source_dict(const char* source_vocab);
    void load_target_dict(const char* target_vocab);

    inline int source_dict_size() {
        return (int)_rev_source_dict_vec.size();
    }

    inline int target_dict_size() {
        return (int)_rev_target_dict_vec.size();
    }

    void load_all_data(const char* source_file,
            const char* target_file,
            const int batch_size = 32,
            const int max_source_len = 50,
            const int max_target_len = 50);

    /*
     * @brief get data for training
     * the caller should make sure the input has enough memory
     * (i.e., setting a larger enough shape and then malloc_all_data)
     *
     * logic inside this routine:
     *    1) pick a bucket according to buckets_scale
     *    2) randomly pick batch_size examples from this bucket
     *    3) pad training pairs to
     *
     * @param [in] batch_size
     * @param [out] encoder_input in shape seq_length * batch, A B C
     * @param [out] decoder_input in shape seq_length * batch, GO W X Y Z
     * @param [out] decoder_target in shape seq_length * batch, W X Y Z EOS
     *
     * @return void
     * @retval none
     *
     */
    void get_batch(
            Blob* encoder_input,
            Blob* decoder_input,
            Blob* decoder_target);

    // for debug purpose
    void display_all_data();
private:

    // data structure: _all_data really stores all the data loaded from training files
    // _buckets is the user specified bucket list
    // _data_set size is same as _buckets, each is an array of index of training examples
    // that should stay in that bucket
    std::vector<std::shared_ptr<TrainPair> > _all_data;
    int _max_source_len;
    int _max_target_len;
    int _batch_size;

    std::shared_ptr<size_t> _example_idx;
    // cursor points to _example_idx
    size_t _cursor;

    // prefetch more batches, then, sort and return a batch
    std::vector<size_t> _prefetched_examples;
    // prefetch this much batch examples
    static const size_t prefetch_batch_count = 20;
    //cursor points to prefetched_examples
    size_t _prefetch_cursor;

    bool prefetch();
    void prepare_data_set();

    void load_dict(const char* vocab_file,
            std::unordered_map<std::string, int>& dict,
            std::vector<std::string>& rev_dict);

    void str_to_ids(const std::string& str,
            std::vector<int>& result,
            const std::unordered_map<std::string, int>& dict);

    std::unordered_map<std::string, int> _source_dict_map;
    std::unordered_map<std::string, int> _target_dict_map;

    std::vector<std::string> _rev_source_dict_vec;
    std::vector<std::string> _rev_target_dict_vec;

    // please refer to tensorflow seq2seq tutorial for the details
    std::string _source_vocab;  // source vocabulary file
    std::string _target_vocab;  // target vocabulary file
    std::string _source_file;   // source training file
    std::string _target_file;   // target training file
private:
    DISALLOW_COPY_AND_ASSIGN(DataReader);
};
} // namespace seq2seq

#endif
