import torch
import clip
from PIL import Image

# --- Load the CLIP model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Load your test image ---
image_path = "test.jpg"  # put your own image file here
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# --- Enter a text query ---
text_query = ["Car"]  # you can try "winter aerial photo", etc.v
text = clip.tokenize(text_query).to(device)

# --- Get embeddings ---
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# --- Compute similarity ---
similarity = (image_features @ text_features.T).squeeze().item()

print(f"Similarity between image and '{text_query[0]}': {similarity:.4f}")
