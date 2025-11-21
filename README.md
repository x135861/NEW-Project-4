# Archive Media Search Tool

A local software tool for searching archived video and image content.  
The tool allows users to search for objects, scenes, or other attributes (e.g., "red car", "winter aerial photo") in large media archives while keeping all data fully local.  

---

## Features

- **Local Processing**: All media is processed and indexed on the local server; no data is sent externally.  
- **Keyword Search**: Search archives using natural language text queries.  
- **Supports Multiple File Formats**:
  - Images: JPG, PNG, TIFF, BMP, GIF, WEBP, HEIC
  - Videos: MP4, MOV, AVI, MKV, WMV, FLV, MPEG
- **Handles Low-Contrast / Grayish Images**: Preprocessing improves feature extraction for challenging imagery.  
- **Flexible Output**: Search results can be returned as file paths, thumbnails, or preview links.  

---

## Technology Stack

- **Backend**: Python, FastAPI / Flask  
- **Model & AI**: PyTorch, OpenAI CLIP / OpenCLIP, optional YOLO object detection  
- **Vector Database**: FAISS for similarity search  
- **Image & Video Processing**: OpenCV, Pillow, FFmpeg  
- **Frontend**: Web UI (React/Vue + Tailwind) or Desktop GUI (PyQt5 / PySide6)  

---

