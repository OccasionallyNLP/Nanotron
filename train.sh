CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 run_train_ok.py --config-file examples/config_smollm_qwen_unimax.yaml
#CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 run_train.py --config-file examples/config_smollm_qwen_unimax.yaml
