import os
import numpy as np
from PIL import Image
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import cv2


model = SentenceTransformer("clip-ViT-B-32")
index_folder = "index"

image_index = faiss.read_index(os.path.join(index_folder, "archive.index"))
with open(os.path.join(index_folder, "paths.txt"), "r") as f:
    image_paths = [line.strip() for line in f.readlines()]

video_index = faiss.read_index(os.path.join(index_folder, "video.index"))
with open(os.path.join(index_folder, "video_frames.pkl"), "rb") as f:
    video_frames = pickle.load(f)

def encode_text(query):
    return model.encode(query, convert_to_tensor=False, normalize_embeddings=True).astype("float32")

def filename_search(query, folder="data/images", max_results=5):
    query = query.lower()
    results = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png")) and query in f.lower():
            results.append(os.path.join(folder, f))
        if len(results) >= max_results:
            break
    return results

def search_video_frames(query, top_k=5):
    query_vec = encode_text(query).reshape(1, -1)
    D, I = video_index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(video_frames):
            video_name, ts = video_frames[idx]
            cap = cv2.VideoCapture(os.path.join("data/videos", video_name))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * cap.get(cv2.CAP_PROP_FPS)))
            success, frame = cap.read()
            if success:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results.append((img, video_name, ts, dist))
            cap.release()
    return results

def search(query, image_folder="data/images", video_folder="data/videos", max_results=5):
    query_vec = encode_text(query).reshape(1, -1)
    D, I = image_index.search(query_vec, max_results)

    image_results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(image_paths) and dist < 1.0:
            image_results.append(image_paths[idx])

    if not image_results:
        image_results = filename_search(query, image_folder, max_results)

    video_results = search_video_frames(query, top_k=max_results)

    return image_results, video_results
