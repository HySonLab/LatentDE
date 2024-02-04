config_file="$1"
devices="$2"
prefix="$3"
dataset="$4"
expected_kl="${5:-20}"
batch_size="${6:-64}"
dirpath=$(realpath $(dirname $0))
csv_file="${dirpath}/preprocessed_data/${dataset}/${dataset}.csv"
epochs=130
seed=42

python scripts/train_vae.py $config_file --csv_file=$csv_file \
		     --expected_kl=$expected_kl --batch_size=$batch_size \
		     --devices=$devices --epochs=$epochs \
		     --seed=$seed --prefix=$prefix