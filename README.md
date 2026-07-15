# 📷 Privacy-First Smart Photo Search (`On-Device AI Assistant`)

[![OSDHack 2026 Submission](https://img.shields.io/badge/OSDHack-2026_Submission-blueviolet?style=for-the-badge)](https://github.com/kyukoten/OSDHACK-2026)
[![Execution: 100% On-Device](https://img.shields.io/badge/Execution-100%25_Local_AI-2ea44f?style=for-the-badge)](./LOCAL_AI_VERIFICATION.md)
[![Model: CLIP ViT--B/32](https://img.shields.io/badge/Model-OpenAI_CLIP_ViT--B%2F32-009688?style=for-the-badge)](./TECHNICAL_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](./ATTRIBUTION.md)

An open-source, offline-first native desktop search assistant engineered for **OSDHack 2026**. This application empowers users to search through personal image directories using natural language semantic queries (e.g., *"a dog playing in the park"*, *"sunset by the beach"*, *"red sports car"*)—**without transmitting a single pixel, file name, or search query to external cloud servers.**

---

## 📚 Hackathon Submission Navigation & Documentation

We have prepared comprehensive, engineering-grade documentation covering every mandatory and recommended evaluation criteria requested by the OSDHack 2026 committee:

| Submission Requirement | Status | Document Link & Overview |
| **System Architecture** *(Recommended)* | ✅ Included | **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — Mermaid system diagrams, Vision/Text Transformer pipeline, multi-threaded I/O flow, and key design decisions. |
| **Technical Report** *(Recommended)* | ✅ Included | **[`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md)** — Live empirical benchmarks, parameter counts (`~151.3M`), latency (`16.9 µs` search), and sub-1GB peak RAM profiles. |
| **Local AI Verification** *(Recommended)*| ✅ Included | **[`LOCAL_AI_VERIFICATION.md`](./LOCAL_AI_VERIFICATION.md)** — Air-gapped guarantees, offline execution matrix, and step-by-step verification protocols. |
| **Empirical Evaluation** *(Recommended)* | ✅ Included | **[`EVALUATION.md`](./EVALUATION.md)** — Top-1 exact ground-truth accuracy metrics (`60% - 100%`), baseline comparisons vs OS/Cloud search, and failure modes. |
| **Privacy & Safety** *(Recommended)* | ✅ Included | **[`PRIVACY_SAFETY.md`](./PRIVACY_SAFETY.md)** — Data governance, read-only OS permissions, vector index storage footprint (`~2KB/img`), and risk mitigations. |
| **Attribution & Licenses** *(Recommended)*| ✅ Included | **[`ATTRIBUTION.md`](./ATTRIBUTION.md)** — Complete recognition of `openai/clip-vit-base-patch32`, PyTorch, Streamlit, Pillow, and Unsplash datasets. |

---

## ⚡ Key Features

* 🔒 **100% Air-Gapped & Offline Execution:** All neural network inference executes locally using your machine's CPU/GPU/NPU. Once initialized, the app works without any internet connection.
* 🧠 **Multimodal Semantic Understanding:** Powered by `OpenAI CLIP (ViT-Base-Patch32)`, mapping text prompts and raster images into a shared `512-dimensional` vector space.
* ⚡ **Instant Precomputed Indexing (`.photo_index.pt`):** Multi-threaded worker pools (`ThreadPoolExecutor`) ingest photos in parallel, serializing L2-normalized feature tensors directly to disk for $O(1)$ instant startup loading.
* 🎛️ **Interactive Strictness Threshold Slider:** Dynamically filter top-ranked results (`15% - 50%`) to eliminate low-confidence noise and customize precision versus recall in real time.
* 💅 **Native App Polish:** Custom CSS injection transforms standard Streamlit elements into a sleek desktop gallery with card hover micro-animations and confidence badges.

---

## 🛠️ Clear Setup Instructions & Dependencies

### Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux
* **Python Version:** Python `3.10`, `3.11`, or `3.12` installed and available on `PATH`

### 1. Clone the Repository
```bash
git clone https://github.com/kyukoten/OSDHACK-2026.git
cd OSDHACK-2026
```
*(Or navigate to your local workspace directory `privacy-photo-search`)*

### 2. Create & Activate Virtual Environment (`Recommended`)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux (Bash)
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```
*Core Dependencies listed in `requirements.txt`:*
* `torch` (PyTorch core engine & tensor operations)
* `torchvision` (Vision transforms & utility hooks)
* `transformers` (Hugging Face `CLIPModel` and `CLIPProcessor`)
* `pillow` (Raster image decoding across JPG, PNG, WEBP, BMP, TIFF)
* `streamlit` (Responsive desktop UI dashboard and custom styling)

---

## 🚀 Run Commands & Steps to Reproduce

### Step 1: Download Ground-Truth Sample Images
To instantly test the system right out of the box with diverse test photos, run our automated sample downloader:
```bash
python download_samples.py
```
*Output: Downloads 5 uncompressed public domain photos into the `./sample_images` folder (`dog_in_park.jpg`, `beach_sunset.jpg`, `red_sports_car.jpg`, `cat_sleeping.jpg`, `coffee_cup_on_desk.jpg`).*

### Step 2: Launch the Desktop Application
```bash
streamlit run app.py
```
*Your default web browser will automatically open to `http://localhost:8501` showing the native app interface.*

### Step 3: Configure Local Directory & Build Index
1. Click on the **⚙️ Engine & Database** tab at the top of the interface.
2. Verify or enter your local directory path in the input box (default: `./sample_images`).
3. Observe the system automatically scan the directory using 4 concurrent threads, extract `512-dim` visual feature embeddings, and save `.photo_index.pt` inside the target directory.
4. *Stats displayed:* **Images Found** count and **Index Status (`Active`)**.

### Step 4: Execute Natural Language Search (`Sample Inputs & Expected Outputs`)
Switch to the **🔍 Search Gallery** tab and try any of the following sample queries:

| Sample Input Query | Strictness Slider | Expected Output & Behavior |
| :--- | :---: | :--- |
| `"a dog playing in the park"` | `25%` | Renders **`dog_in_park.jpg`** in the grid with a **`~26% Match`** confidence badge and file path caption. |
| `"cat sleeping peacefully"` | `24%` | Renders **`cat_sleeping.jpg`** with a **`~25% Match`** confidence badge. |
| `"coffee cup on desk"` | `24%` | Renders **`coffee_cup_on_desk.jpg`** accurately distinguishing workspace items from outdoor scenes. |
| `"red sports car"` | `26%` | Renders high-contrast red automotive illustrations or **`red_sports_car.jpg`**. |
| *"Unrelated Prompt e.g., 'spaceship in galaxy'"* | `35%` | Renders yellow warning box: *"No high-confidence matches found... Try lowering the strictness slider or changing the prompt."* |

---

## 🔬 Run Quantitative Evaluation Harness (`Optional Benchmark`)

To independently verify our latency, peak RAM footprint, and top-1 retrieval accuracy reported in the Technical Report, execute our automated evaluation suite directly from your terminal:
```bash
python benchmark_eval.py
```
*This script will load the `openai/clip-vit-base-patch32` model, measure exact loading latency, average extraction milliseconds per image across all photos in `./sample_images`, run 1,000 vector dot-product search loops, and output exact memory consumption and ground-truth retrieval scores.*

---

## 📂 Project Directory Structure

```text
OSDHACK-2026/
├── app.py                      # Main Streamlit application & multi-threaded indexing logic
├── download_samples.py         # Script to download 5 ground-truth benchmark photos
├── benchmark_eval.py           # Automated latency, memory & accuracy evaluation suite
├── requirements.txt            # Minimal Python dependency list
├── README.md                   # Primary documentation & setup instructions (This file)
├── ARCHITECTURE.md             # System diagram, CLIP ViT-B/32 pipeline & engineering choices
├── TECHNICAL_REPORT.md         # Empirical latency, model size (~151M), and memory profiles
├── LOCAL_AI_VERIFICATION.md    # Air-gapped verification audit and zero cloud dependencies
├── EVALUATION.md               # Retrieval accuracy tables, baseline comparison & failure cases
├── PRIVACY_SAFETY.md           # Data handling policies, read-only permissions & risk analysis
├── ATTRIBUTION.md              # Open-source model, library, dataset & license attributions
└── sample_images/              # Directory containing user photos and .photo_index.pt
```

---

## 🏁 Built with ❤️ for OSDHack 2026
*Empowering users with private, zero-cloud artificial intelligence on consumer hardware.*