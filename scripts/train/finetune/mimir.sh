#!/bin/bash
# per https://arxiv.org/pdf/2411.15124,
# the effective batch size is 128 using a single GPU, since we simulate it with:
# gradient_accumulation_steps 16 * per_device_train_batch_size 1 * num_processes 8 = 128
accelerate launch \
    --mixed_precision bf16 \
    --num_processes $1 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --model_name_or_path $2/Mimir-1B-stage1-150K \
    --tokenizer_name $2/Mimir-1B-stage1-150K \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir $2/Mimir-1B-stage1-150K-sft \
    --with_tracking \
    --report_to aim \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list $3/tulu-3-sft-mixture 1.0 \
    --dataset_mix_dir $2/Mimir-1B-stage1-150K-sft \
    --exp_name Mimir-1B-stage1-150K-sft \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --timeout 86400 \
    --hf_entity local \
    --checkpointing_steps 100 \
    --keep_last_n_checkpoints 1000 \
    --aim_repo $3/open-instruct \
    --seed 9313
