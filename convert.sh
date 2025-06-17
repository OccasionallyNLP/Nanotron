for NAME in unimax token_based order_matters
do
torchrun --nproc_per_node=1 -m examples.llama.convert_nanotron_to_hf --checkpoint_path ../checkpoints/1b/${NAME}/12500 --save_path ../hf_checkpoints/1b/${NAME}/12500 --tokenizer_name ./Qwen3-8B
done


