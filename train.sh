export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export EXP='flir_lora'
export OUTPUT_DIR="finetune/$EXP"
export DATASET_DIR="/media/dataset/flir/FLIR_ADAS_1_3" # path to FLIR dataset

accelerate launch --multi-gpu train_text_to_image_lora_flir.py \
  --dataset_dir=$DATASET_DIR \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataloader_num_workers=8 \
  --train_batch_size=3 \
  --run_name=$EXP \
  --num_train_epochs=50 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --checkpointing_steps=1850 \
  --lr_scheduler="cosine" \
  --resolution=512 \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --rank=4 \
  --validation_prompt="" \
  --seed=42


