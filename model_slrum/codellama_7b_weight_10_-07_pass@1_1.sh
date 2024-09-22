cd $current_dir

CUDA_VISIBLE_DEVICES=0 python3 ./CodeGen-USCD/main.py \
    --use_auth_token \
    --mean_num 0.0050 \
    --filter_num 0.666667 \
    --num_beams 1 \
    --src_weights 1.0 -0.7 \
    --model $model_dir \
    --tasks humaneval \
    --temperature 0.8 \
    --n_samples 1 \
    --top_p 0.95 \
    --allow_code_execution \
    --metric_output_path ./CodeGen-USCD/metric_output/codellama_7b_weight_10_-07_pass@1_1_results.json \
    --save_generations_path ./CodeGen-USCD/generation_output/codellama_7b_weight_10_-07_pass@1_1_generations.json
    