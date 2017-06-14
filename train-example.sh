set -ex
me=$(basename $0|cut -d. -f1)
device_id=1
data_set=data/wangfeng
save_model_dir="${data_set}_${me}_${device_id}"
mkdir -p "${save_model_dir}"

./bin/seq2seq -device ${device_id} \
    -train_data_dir ${data_set} \
    -save_model_dir "${save_model_dir}" \
    -emb_size 620 \
    -hidden_size 1000 \
    -max_iter 20000000 \
    -save_per_iter 68000 \
    -batch_size 32 \
    -checkpoint_per_iter 1000 \
    -lr 1.0 \
    -lr_decay 0.998 2> "${save_model_dir}.log" &
