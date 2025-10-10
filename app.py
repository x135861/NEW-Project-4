import streamlit as st
import json
import torch
import faiss
import numpy as np
import open_clip
from PIL import Image

# --- Load model and data ---
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model_and_index():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device).eval()
    index = faiss.read_index("index.faiss")
    with open("meta.json") as f:
        meta = json.load(f)
    return model, preprocess, tokenizer, index, meta

model, preprocess, tokenizer, index, meta = load_model_and_index()

# --- Search functions ---
def search_text(query, k=5):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    feat_np = feat.cpu().numpy().astype('float32')
    scores, ids = index.search(feat_np, k)
    return [(meta[i]["path"], float(scores[0][j])) for j, i in enumerate(ids[0])]

def search_image(img, k=5):
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    feat_np = feat.cpu().numpy().astype('float32')
    scores, ids = index.search(feat_np, k)
    return [(meta[i]["path"], float(scores[0][j])) for j, i in enumerate(ids[0])]

# --- UI ---
st.set_page_config(page_title="CLIP Search", layout="centered")
st.title("🔎 Search Tool Demo")


st.sidebar.title("Search Options")
search_mode = st.sidebar.radio("Search by:", ["Text", "Image"])

# Initialize results safely
results = []

# --- Text search ---
if search_mode == "Text":
    query = st.text_input("Enter your text query:")
    if st.button("Search") and query:
        results = search_text(query)

# --- Image search ---
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Query Image", width=300)
        if st.button("Search"):
            results = search_image(img)

# --- Display results ---
if results:
    st.write("### Results:")

    # Display results in a responsive grid
    max_cols = 3
    cols = st.columns(min(len(results), max_cols))
    
    for idx, (path, score) in enumerate(results):
        col = cols[idx % max_cols]
        caption = f"**File:** {path.split('/')[-1]}  \n**Score:** {score:.3f}"
        col.image(path, caption=caption, use_column_width=True)
else:
    st.write("No results yet. Perform a search to see results.")
