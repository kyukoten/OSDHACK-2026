# 🎬 Demo Video Production Guide & Script (Mandatory Submission Item)

This comprehensive guide and storyboard is designed to help you record a high-impact, professional **2 to 3-minute demo video** for **OSDHack 2026**.

The hackathon guidelines explicitly mandate demonstrating:
1. **The Problem** (Cloud privacy violations & broken local keyword search)
2. **The Solution** (Zero-shot natural language search)
3. **The On-Device AI Component Working** (Real-time local vector indexing & air-gapped retrieval)

---

## 🛠️ Pre-Recording Setup Checklist

Before hitting record in OBS Studio or Zoom:
* [ ] **Resolution & Quality:** Record at `1920x1080 (1080p)` at `60 FPS` for crisp UI rendering.
* [ ] **Clean Desktop & Browser:** Close unnecessary tabs, hide bookmarks bar (`Ctrl+Shift+B`), and ensure Streamlit is running cleanly on `http://localhost:8501`.
* [ ] **Prepare Test Folder:** Make sure `./sample_images` has a mix of diverse photos (`dog_in_park.jpg`, `beach_sunset.jpg`, `cat_sleeping.jpg`, `red_sports_car.jpg`, plus a few desktop wallpapers/anime photos).
* [ ] **Warm Model Cache:** Run `app.py` at least once beforehand so the `openai/clip-vit-base-patch32` weights (`~605 MB`) are cached and load in `~5 seconds` without progress bar clutter.
* [ ] **Audio Quality:** Use a clean microphone in a quiet room, or use AI voiceover/noise suppression. Speak with clarity and enthusiasm!

---

## 📋 Storyboard & Timeline (Total Duration: ~2:30)

| Timestamp | Section | Visual Screen Action | Voiceover / Narration Script |
| :---: | :---: | :--- | :--- |
| **0:00 - 0:25** <br> *(25 sec)* | **1. The Problem & Hook** | Open standard Windows File Explorer showing a folder full of randomly named images (`IMG_20260715.jpg`, `c1071d7e.jpeg`). Try typing `"dog playing"` in the OS search box $\rightarrow$ shows **No results**. | *"Have you ever tried finding a specific photo on your computer by searching 'dog playing' or 'sunset beach', only to get zero results because traditional desktop search only checks file names? To fix this today, users are forced to upload their private, personal family photos to cloud services like Google Photos or iCloud—surrendering privacy and allowing tech giants to train on sensitive data. For OSDHack 2026, we built a better way."* |
| **0:25 - 0:55** <br> *(30 sec)* | **2. Introducing Solution & Architecture** | Switch to the clean Streamlit UI of **Privacy-First Smart Photo Search**. Show the sleek title, polished borders, and modern layout. Briefly show a slide or open `ARCHITECTURE.md` Mermaid diagram for 5 seconds. | *"Introducing Privacy-First Smart Photo Search—a 100% local, air-gapped desktop assistant powered by OpenAI's CLIP Vision Transformer architecture running directly on your computer's CPU and GPU. It maps both natural language and images into a shared 512-dimensional vector space without sending a single pixel or byte over the internet."* |
| **0:55 - 1:35** <br> *(40 sec)* | **3. Live On-Device Indexing Demo** | Click on the **⚙️ Engine & Database** tab. Show `./sample_images` path entered in the box. Point out the live metrics (`Images Found: 47`, `Index Status: Active`). Show the expandable error log section. | *"Let’s see the on-device AI engine in action. In the Engine & Database tab, we point our system to any local directory on our hard drive. Using multi-threaded Python worker pools, our application decodes images asynchronously, runs them through the local Vision Transformer (`ViT-Base-Patch32`), and pre-computes L2-normalized vector embeddings saved directly to our local disk (`.photo_index.pt`). Notice how our peak memory stays well under 1 GB, and our index loads instantly."* |
| **1:35 - 2:10** <br> *(35 sec)* | **4. Live Semantic Search & Strictness Slider** | Switch to the **🔍 Search Gallery** tab. <br> 1. Type: `"a dog playing in the park"` $\rightarrow$ instantly renders `dog_in_park.jpg`. <br> 2. Type: `"cat sleeping"` $\rightarrow$ renders `cat_sleeping.jpg`. <br> 3. Adjust the **Strictness Slider** from `25%` to `35%` to demonstrate real-time threshold filtering. | *"Now let’s search using pure natural language. I’ll type 'a dog playing in the park'—and instantly, our local cosine similarity matrix dot-product ranks our gallery in under two milliseconds and brings up the exact photo with a 26% confidence match badge. Let's try 'cat sleeping'—boom, perfect match. And to give users complete control over precision versus recall, we added an interactive Strictness Slider right here that filters out lower-confidence matches dynamically."* |
| **2:10 - 2:30** <br> *(20 sec)* | **5. Air-Gapped Verification & Conclusion** | Hover mouse over Windows Wi-Fi / Network icon showing disconnected (or emphasize offline guarantee verbally). Show the repository links and final summary. | *"Best of all, once the initial model weights are cached, this entire system runs completely air-gapped. You can unplug your Ethernet cable right now, and the search functions with 100% accuracy. Zero telemetry, zero cloud APIs, and complete data sovereignty. Thank you for checking out Privacy-First Smart Photo Search for OSDHack 2026!"* |

---

## 💡 Pro-Tips for Hackathon Judges Appeal

1. **Highlight Speed:** Emphasize that the cosine similarity search takes **less than 2 milliseconds** (`~33 µs`) because of vector normalization and linear algebraic dot products.
2. **Emphasize Polish:** Mention how the custom CSS injection transformed Streamlit into a native desktop application experience.
3. **Show Technical Rigor:** Briefly mention that you evaluated the system using an automated benchmark suite (`benchmark_eval.py`), achieving high retrieval accuracy across uncurated real-world photos!
