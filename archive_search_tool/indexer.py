import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

# Model for image embeddings
model = SentenceTransformer("clip-ViT-B-32")  

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    embedding = model.encode(image, convert_to_tensor=False, normalize_embeddings=True)
    return embedding

def build_index(image_folder="data/images", index_folder="index"):
    os.makedirs(index_folder, exist_ok=True)
    image_paths, embeddings = [], []

    for file in os.listdir(image_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(image_folder, file)
            try:
                emb = extract_features(path)
                embeddings.append(emb)
                image_paths.append(path)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_folder, "archive.index"))

    with open(os.path.join(index_folder, "paths.txt"), "w") as f:
        f.write("\n".join(image_paths))

    print(f"✅ Indexed {len(image_paths)} images successfully!")

if __name__ == "__main__":
    build_index()
