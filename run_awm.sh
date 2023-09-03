# torchrun --nproc_per_node 1 example_text_completion.py \
<<<<<<< HEAD
=======
#     --ckpt_dir /root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat/snapshots/2abbae1937452ebd4eecb63113a87feacd6f13ac \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4 \
#     --temperature 0.8 \
#     --watermark None \
#     --hashing_schema lefthash


# torchrun --nproc_per_node 1 example_text_completion.py \
#     --ckpt_dir /root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat/snapshots/2abbae1937452ebd4eecb63113a87feacd6f13ac \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4 \
#     --temperature 0.8 \
#     --watermark aaronson \
#     --hashing_schema lefthash


torchrun --nproc_per_node 1 example_chat_completion_siyuan.py \
    --ckpt_dir /root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat/snapshots/2abbae1937452ebd4eecb63113a87feacd6f13ac \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 8 \
    --temperature 1.0 \
    --watermark aaronson \
    --hashing_schema lefthash

# torchrun --nproc_per_node 1 example_chat_completion_siyuan.py \
#     --ckpt_dir /root/autodl-tmp/huggingface/hub/models--meta-llama--Llama-2-7b-chat/snapshots/2abbae1937452ebd4eecb63113a87feacd6f13ac \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 8 \
#     --temperature 1.0 \
#     --watermark none \
#     --hashing_schema lefthash


# torchrun --nproc_per_node 1 example_text_completion.py \
>>>>>>> refs/remotes/origin/main
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
<<<<<<< HEAD
#     --hashing_schema lefthash

torchrun --nproc_per_node 2 example_text_completion.py \
    --ckpt_dir /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat/snapshots/8ebc6a0ac2e4a781c31cb4ad395b1c26c5158c76 \
    --tokenizer_path /home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat/snapshots/8ebc6a0ac2e4a781c31cb4ad395b1c26c5158c76 \
    --max_seq_len 128 --max_batch_size 4 \
    --temperature 0.6 \
    --watermark aaronson \
    --hashing_schema lefthash
=======
#     --hashing_schema lefthash
>>>>>>> refs/remotes/origin/main
