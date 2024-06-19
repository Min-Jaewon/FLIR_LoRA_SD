from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionPipeline
import torch
from torchvision.utils import save_image
from huggingface_hub import model_info


device='cuda'
for seed in range(42, 48):
    torch.manual_seed(seed)
    
    pipeline_lora = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", custom_pipeline="hyoungwoncho/sd_perturbed_attention_guidance", torch_dtype=torch.float16).to(device)
    pipeline_lora.load_lora_weights("lora_weight", weight_name="pytorch_lora_weights.safetensors")

    image_org = pipeline_lora("",
                                width=512,
                                height=512,
                                num_inference_steps=50,
                                guidance_scale=0.0).images[0]
    image_org.save(f'result/org_{seed}.png')

    image_lora = pipeline_lora("",
                                width=512,
                                height=512,
                                num_inference_steps=50,
                                guidance_scale=0.0,
                                pag_scale=5.0,
                                pag_applied_layers_index=['m0']).images[0]
    image_lora.save(f'result/pag_{seed}.png')