export CUDA_VISIBLE_DEVICES=0,1
# export OPENAI_API_KEY="d8e0a18a-776f-46f9-9a2b-4585286e47ca"
# export BASE_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
# export MODEL="deepseek-v3-250324"

# echo $OPENAI_API_KEY

python -m eval.MATH.run_eval \
    --data_dir data/math \
    --save_dir results/llama3-sft-noicl/math \
    --n_shot 4 \
    --use_vllm \
    --model_name_or_path ../llama3-8b-segment

