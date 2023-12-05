GPU=1
for data in xsum cnn_dailymail; do
for shots in 0 3 5; do
CUDA_VISIBLE_DEVICES=${GPU} python -u benchmarks/benchmark_summarization.py \
--model-name EleutherAI/pythia-2.8b-v0 \
--input_path data/${data}_${shots}shot.jsonl \
--sample_num 200
done
done