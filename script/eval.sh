GPU=$1
data=$2
CUDA_VISIBLE_DEVICES=${GPU} python -u benchmarks/benchmark_summarization.py \
--model-name state-spaces/mamba-2.8b \
--input_path ${data} \
--sample_num 100