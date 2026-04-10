MODEL=$1
DATA_DIR=$2
DEVICES=$3

export CUDA_VISIBLE_DEVICES=$DEVICES

python -m eval.mmlu.run_eval \
    --data_dir $DATA_DIR \
    --save_dir results/$MODEL/$DATA_DIR \
    --model_name_or_path $MODEL \
    --ntrain 0 \
    --cot \
    --model_type vllm \
    --seg