export WANDB_MODE="offline"
export MODEL_PATH="pretrained"
export CONFIG_PATH="pretrained"
export TYPE="i2v"
export DATASET_PATH="datasets/youtube_filtered_part1/total_train_data_novico.txt"
export OUTPUT_PATH="bindyouravatar_final"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_TIMEOUT=600
export NCCL_SOCKET_TIMEOUT=600
# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file util/deepspeed_configs/accelerate_config_machine_single.yaml \
  train.py \
  --config_path $CONFIG_PATH \
  --dataloader_num_workers 8 \
  --pretrained_model_name_or_path $MODEL_PATH \
  --instance_data_root $DATASET_PATH \
  --validation_prompt "The video features a woman standing next to an airplane, engaged in a conversation on her cell phone. She is wearing sunglasses and a black top, and she appears to be talking seriously. The airplane has a green stripe running along its side, and there is a large engine visible behind her. The woman seems to be standing near the entrance of the airplane, possibly preparing to board or just having disembarked. The setting suggests that she might be at an airport or a private airfield. The overall atmosphere of the video is professional and focused, with the woman's attire and the presence of the airplane indicating a business or travel context." \
  --validation_images "asserts/example_images/5.png" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $OUTPUT_PATH \
  --height 480 \
  --width 720 \
  --max_num_frames 49 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --checkpointing_steps 100 \
  --num_train_epochs 4 \
  --lr_warmup_steps 100 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to wandb \
  --sample_stride 1 \
  --skip_frames_start 2 \
  --skip_frames_end 2 \
  --miss_tolerance 1 \
  --min_distance 0 \
  --min_frames 1 \
  --max_frames 1 \
  --cross_attn_interval 2 \
  --cross_attn_dim_head 128 \
  --cross_attn_num_heads 16 \
  --LFE_id_dim 1280 \
  --LFE_vit_dim 1024 \
  --LFE_depth 10 \
  --LFE_dim_head 64 \
  --LFE_num_heads 16 \
  --LFE_num_id_token 5 \
  --LFE_num_querie 32 \
  --LFE_output_dim 2048 \
  --LFE_ff_mult 4 \
  --LFE_num_scale 5 \
  --local_face_scale 1.0 \
  --is_train_face \
  --is_align_face \
  --train_type $TYPE \
  --is_shuffle_data \
  --enable_mask_loss \
  --mask_prob 0.2 \
  --is_train_audio \
  --tracker_name "bindyouravatar-final" \
  --gradient_checkpointing \
  --resume_from_checkpoint="latest" \
  --router_loss_weight 1 \
  --consistency_loss_weight 8 \
  --temporal_diff_loss_weight 0.002 \
  --spatial_diff_loss_weight 0.0009 \
  --spatial_dist_loss_weight 10 \
  --id_dist_loss_weight 10 \
  --is_train_lora \
  --load_pretrained_module  \
  --load_pretrained_modules_list "audio_module" "face_module" "lora_weight"    \
  --load_pretrained_modules_list_path "audio_modules_2_1_pre.pt" "face_modules_v2_1_pre.pt" "pytorch_lora_weights_v2_1.safetensors" \
  --unfreeze_modules "audio_model.mute_learnable_tokens" "perceiver_cross_attention" "audio_model.layers" "router" \
  --freeze_modules  "no_freeze" \
  --is_accelerator_state_dict \
  --is_teacher_forcing \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --index_mask_drop_prob 0.2 \
  --step_timeout 300 \
  --routing_logits_zeros_prob 0.2 





  # --lr_scheduler constant
  # --low_vram \
  # --resume_from_checkpoint="latest" \
  # --is_single_face \
  # --is_validation \
  # --is_kps \
  # --pretrained_weight "checkpoint-1250" \
  # --is_diff_lr \
  # --low_vram \
  # --is_cross_face
  # --enable_slicing \
  # --enable_tiling \
  # --use_ema
  #  --fps 8 \


