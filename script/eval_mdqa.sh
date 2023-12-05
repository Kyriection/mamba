
GPU=1
for idx in 1; do
CUDA_VISIBLE_DEVICES=${GPU} python -u benchmarks/benchmark_mdqa.py \
    --model-name state-spaces/mamba-2.8b \
    --answer_idx ${idx} \
    --output_path qa_results/MDQA_${idx}answer-mamba_2.8b.jsonl \
    --sample_num 200
python -u eval_qa_response.py --input-path qa_results/MDQA_${idx}answer-mamba_2.8b.jsonl
done 