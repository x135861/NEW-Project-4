import streamlit as st
from search_engine import search
from PIL import Image

st.set_page_config(page_title="Archive Image & Video Search", layout="wide")
st.title("📸 Search Tool")

# Input
query = st.text_input("Enter your search keyword:", "")
max_results = st.slider("Max results to display:", 1, 20, 5)

if st.button("🔍 Search") and query.strip():
    st.write(f"### Searching for '{query}'...")
    image_results, video_results = search(query.strip(), max_results=max_results)

    # Display images
    if image_results:
        st.write("### Images:")
        for path in image_results:
            st.image(Image.open(path), caption=path, width=300)  # fixed smaller width
    else:
        st.write("No matching images found.")

    # Video frames (optional)
    if video_results:
        st.write("### Video frames:")
        for img, video, ts, score in video_results:
            st.image(img, caption=f"{video} at {ts:.2f}s (score {score:.2f})", width=300)
    else:
        st.write("No matching video frames found.")
