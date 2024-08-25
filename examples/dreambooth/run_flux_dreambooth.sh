export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="/home/niels/python_projects/Foto's_Niels"
export OUTPUT_DIR="trained-flux-lora"

CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of niels" \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-6 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1200 \
  --seed="0" \
  --push_to_hub