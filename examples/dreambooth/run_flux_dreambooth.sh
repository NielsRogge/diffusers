export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="/home/niels/python_projects/Foto's_Niels"
export OUTPUT_DIR="trained-flux-lora"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks niels" \
  --optimizer="prodigy" \
  --weighting_scheme="none"\
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" \
  --push_to_hub