# torchrun --nproc_per_node 1 example_text_completion.py \
#     --ckpt_dir llama-2-7b/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4 \
#     --temperature 0.6 \
#     --watermark aaronson \
#     --hashing_schema lefthash

# torchrun --nproc_per_node 1 example_text_completion.py \
#     --ckpt_dir /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235 \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4 \
#     --temperature 0.6 \
#     --watermark aaronson \
#     --hashing_schema lefthash

torchrun --nproc_per_node 2 example_text_completion.py \
    --ckpt_dir /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat/snapshots/8ebc6a0ac2e4a781c31cb4ad395b1c26c5158c76 \
    --tokenizer_path /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat/snapshots/8ebc6a0ac2e4a781c31cb4ad395b1c26c5158c76 \
    --max_seq_len 128 --max_batch_size 4 \
    --temperature 0.6 \
    --watermark aaronson \
    --hashing_schema lefthash