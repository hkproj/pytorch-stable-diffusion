import gradio as gr
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from nanograd.models.stable_diffusion import model_loader, pipeline

DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer_vocab_path = Path("tokenizer_vocab.json")
tokenizer_merges_path = Path("tokenizer_merges.txt")
model_file = Path("v1-5-pruned-emaonly.ckpt")

tokenizer = CLIPTokenizer(str(tokenizer_vocab_path), merges_file=str(tokenizer_merges_path))
models = model_loader.preload_models_from_standard_weights(str(model_file), DEVICE)

def generate_image(prompt, cfg_scale, num_inference_steps, sampler):
    uncond_prompt = ""
    do_cfg = True
    input_image = None
    strength = 0.9
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image = Image.fromarray(output_image)
    return output_image

# Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(label="Prompt", placeholder="A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution")
                cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=100, value=20, step=5)
                sampler = gr.Radio(label="Sampling Method", choices=["ddpm", "Euler a", "Euler", "LMS", "Heun", "DPM2 a", "PLMS"], value="ddpm")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                output_image = gr.Image(label="Output", show_label=False, height=512, width=512)

        generate_btn.click(fn=generate_image, inputs=[prompt_input, cfg_scale, num_inference_steps, sampler], outputs=output_image)

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
