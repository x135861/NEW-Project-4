import os
import numpy as np
from PIL import Image
import cv2
from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("clip-ViT-B-32")
video_folder = "data/videos"
index_folder = "index"
os.makedirs(index_folder, exist_ok=True)

frame_paths = []
embeddings = []
frame_interval = 30  # every N frames

for video in os.listdir(video_folder):
    if not video.lower().endswith(".mp4"):
        continue
    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((224,224))
            emb = model.encode(img, convert_to_tensor=False, normalize_embeddings=True).astype("float32")
            embeddings.append(emb)
            frame_paths.append((video, count / fps))  # store video name + timestamp
        success, frame = cap.read()
        count += 1
    cap.release()

embeddings = np.array(embeddings).astype("float32")
video_index = faiss.IndexFlatL2(embeddings.shape[1])
video_index.add(embeddings)

# Save index and frame metadata
faiss.write_index(video_index, os.path.join(index_folder, "video.index"))
with open(os.path.join(index_folder, "video_frames.pkl"), "wb") as f:
    pickle.dump(frame_paths, f)

print(f"✅ Indexed {len(frame_paths)} video frames successfully!")
