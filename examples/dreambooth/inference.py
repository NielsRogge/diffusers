from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('nielsr/trained-flux-lora', weight_name='pytorch_lora_weights.safetensors')

image = pipeline('a photo of sks niels on the beach',
    height=768,
    width=1360,
    guidance_scale=3.0).images[0]

image.save('sks_niels.png')
