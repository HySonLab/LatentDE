dataset=$1
devices=$2
prefix=$3
model_ckpt_path=$4
oracle_ckpt_path=$5
num_batch=${6:-1}
dec_type=${7:-"rnn"}

dirpath=$(realpath $(dirname $0))

if [[ $dec_type == "rnn"]]; then
	config_file="${dirpath}/scripts/configs/rnn_template.py"
elif [[ $dec_type == "cnn"]]; then
	config_file="${dirpath}/scripts/configs/cnn_template.py"

ref_file="${dirpath}/preprocessed_data/${dataset}/${dataset}_PEX_sequence.txt"

if [[ $dataset == "AMIE" ]]; then
	grad_lr=0015
	scale=4
elif [[ $dataset == "TEM" ]]; then
	grad_lr=0.002
	scale=3
elif [[ $dataset == "LGK" ]]; then
	grad_lr=0.002
	scale=1.5
elif [[ $dataset == "UBE2I" ]]; then
	grad_lr=0.002
	scale=6
fi

output_dir="${dirpath}/exps/results"
steps=500
num_samples=128
beam=1
num_gen=5
num_queries=10
seed=0

expected_kl=20
optim_lr=0.0002
max_epochs=30
patience=8

# python3 scripts/active_optimize.py config_file=$config_file --ref_file=$ref_file \
# 			   --devices=$devices --seed=$seed --model_ckpt_path=$model_ckpt_path \
# 			   --grad_lr=$grad_lr --steps=$steps --num_samples=$num_samples \
# 			   --beam=$beam --num_gen=$num_gen --scale=$scale --oracle_ckpt_path=$oracle_ckpt_path \
# 			   --num_queries=$num_queries --optim_lr=$optim_lr --expected_kl=$expected_kl \
# 			   --max_epochs=$max_epochs --batch=$batch --patience=$patience --prefix=$prefix --eval

for seed in 0 1 2 3 4
do
python3 scripts/active_optimize.py config_file=$config_file --ref_file=$ref_file \
			   --devices=$devices --seed=$seed --model_ckpt_path=$model_ckpt_path \
			   --grad_lr=$grad_lr --steps=$steps --num_samples=$num_samples --output_dir=$output_dir \
			   --beam=$beam --num_gen=$num_gen --scale=$scale --oracle_ckpt_path=$oracle_ckpt_path \
			   --num_queries=$num_queries --optim_lr=$optim_lr --expected_kl=$expected_kl --num_batch=$num_batch \
			   --max_epochs=$max_epochs --batch=$batch --patience=$patience --prefix=$prefix --csv_file=$csv_file --eval
done
