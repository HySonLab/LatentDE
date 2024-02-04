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

if [[ $dataset == "AAV" ]]; then
	lr=0.0015
	scale=3.5
elif [[ $dataset == "avGFP" ]]; then
	lr=0.001
	scale=2
elif [[ $dataset == "AMIE" ]]; then
	lr=0015
	scale=4
elif [[ $dataset == "E4B" ]]; then
	lr=0.002
	scale=3
elif [[ $dataset == "TEM" ]]; then
	lr=0.002
	scale=3
elif [[ $dataset == "LGK" ]]; then
	lr=0.002
	scale=1.5
elif [[ $dataset == "Pab1" ]]; then
	lr=0.001
	scale=2
elif [[ $dataset == "UBE2I" ]]; then
	lr=0.002
	scale=6
fi

output_dir="${dirpath}/exps/results_no_active"
steps=500
num_samples=128
beam=1
num_gen=10
# seed=0

# python3 scripts/optimize.py config_file=$config_file --ref_file=$ref_file \
# 			   --devices=$devices --seed=$seed \
# 			   --model_ckpt_path=$model_ckpt_path \
# 			   --oracle_ckpt_path=$oracle_ckpt_path --lr=$lr --steps=$steps \
# 			   --num_samples=$num_samples --beam=$beam \
# 			   --num_gen=$num_gen --scale=$scale --eval

for seed in 0 1 2 3 4 
do
python3 scripts/optimize.py config_file=$config_file --ref_file=$ref_file \
			   --devices=$devices --seed=$seed --output_dir=$output_dir \
			   --model_ckpt_path=$model_ckpt_path \
			   --oracle_ckpt_path=$oracle_ckpt_path --lr=$lr --steps=$steps \
			   --num_samples=$num_samples --beam=$beam --num_batch=$num_batch \
			   --num_gen=$num_gen --scale=$scale --eval --prefix=$prefix
done
