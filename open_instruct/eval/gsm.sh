MODEL=$1
DATA_DIR=$2
DEVICES=$3

export CUDA_VISIBLE_DEVICES=$DEVICES

echo $DATA_DIR

python -m eval.gsm.run_eval \
    --data_dir $DATA_DIR \
    --save_dir results/$MODEL/tokenseg-$DATA_DIR \
    --model $MODEL \
    --n_shot 8 \
    --use_vllm \
    --stop_at_double_newline
    # --pause
    # --use_chat_format
    # --seg \
    # --k 128
# --seg and --k are used for n-token segmentation
# --use_chat_format is testing for qwen2.5-72b-instruct
# --n_shot default to 8 for qwen2-7b, 4 for qwen2.5-72b
# export OPENAI_API_KEY="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"