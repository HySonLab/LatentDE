dataset=$1
devices=$2
prefix=$3
model_ckpt_path=$4
oracle_ckpt_path=$5
dec_type=${6:-"rnn"}

dirpath=$(realpath $(dirname $0))

csv_file="${dirpath}/preprocessed_data/${dataset}/${dataset}.csv"

if [[ $dec_type == "rnn"]]; then
	config_file="${dirpath}/scripts/configs/rnn_template.py"
elif [[ $dec_type == "cnn"]]; then
	config_file="${dirpath}/scripts/configs/cnn_template.py"

ref_file="${dirpath}/preprocessed_data/${dataset}/${dataset}_PEX_sequence.txt"

batch_size=128
seed=0
is_predicted=1
split="val"


python3 scripts/visualize_latent.py config_file=$config_file --csv_file=$csv_file --ref_file=$ref_file \
			   --devices=$devices --seed=$seed \
			   --model_ckpt_path=$model_ckpt_path \
			   --oracle_ckpt_path=$oracle_ckpt_path \
			   --batch_size=$batch_size --is_predicted=$is_predicted --split=$split
	
