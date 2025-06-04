for NAME in 7000
do
torchrun --nproc_per_node=1 -m examples.llama.convert_nanotron_to_hf --checkpoint_path ../checkpoints/${NAME} --save_path ../hf_checkpoints/${NAME} --tokenizer_name Qwen/Qwen3-8B
done


