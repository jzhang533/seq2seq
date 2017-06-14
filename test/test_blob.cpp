#include <cstring>
#include <unistd.h>
#include <assert.h>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <cuda.h>
#include "all_computes.h"

namespace seq2seq {
void test_blob() {
    Blob blob;
    blob.dim0 = 32;
    blob.dim1 = 128;
    assert(blob.dim0 * blob.dim1 == blob.size());

    blob.malloc_all_data();
    float data[blob.size()];

    for (int i = 0; i < blob.size(); ++i) {
        data[i] = (float)rand() / (float)INT_MAX;
    }
    memcpy(blob.host_data, data, blob.size() * sizeof(float));
    blob.copy_data_to_device();
    memset(blob.host_data, blob.size() * sizeof(float), 0.0);
    blob.copy_data_to_host();
    for (int i = 0; i < blob.size(); ++i) {
        assert(blob.host_data[i] == data[i]);
    }
}
} // namespace seq2seq

int main(int argc, char** argv) {
    srand(time(NULL));
    seq2seq::test_blob();
    return 0;
}
