import torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt_text: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "runwayml/stable-diffusion-v1-5"
    # Updated torch_dtype to dtype
    pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float32)
    pipe = pipe.to(device)

    image = pipe(prompt_text).images[0]
    image.save("output_temple.png")
    print("Image saved successfully as output_temple.png")

if __name__ == "__main__":
    prompt = "A cinematic photograph of an ancient stone temple hidden in a lush rainforest"
    generate_image(prompt)