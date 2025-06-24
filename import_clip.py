import torch
import clip

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

print("Model class:", model.visual.__class__)
print("Output dim:", model.visual.output_dim)
print(model)

