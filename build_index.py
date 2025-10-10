import os, json, torch, faiss, numpy as np
from PIL import Image
import open_clip

# --- Settings ---
image_folder = "images"  # Folder containing your images
index_file = "index.faiss"
meta_file = "meta.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load CLIP model ---
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()

# --- Extract features ---
features = []
meta = []

print("🔍 Extracting features from images...")
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(image_folder, filename)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
            meta.append({"path": path})
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

# --- Build FAISS index ---
print("📦 Building FAISS index...")
features_np = np.concatenate(features, axis=0).astype("float32")
index = faiss.IndexFlatIP(features_np.shape[1])  # Cosine similarity
index.add(features_np)
faiss.write_index(index, index_file)

# --- Save metadata ---
with open(meta_file, "w") as f:
    json.dump(meta, f)

print(f"✅ Index saved to '{index_file}' and metadata to '{meta_file}'")
