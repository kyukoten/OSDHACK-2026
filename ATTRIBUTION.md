# 📜 Open-Source Attributions & Acknowledgments

This document formally recognizes and attributes all pre-trained models, datasets, open-source libraries, frameworks, and foundational work utilized in building **Privacy-First Smart Photo Search** for **OSDHack 2026**.

---

## 1. Pre-Trained Foundational Models

| Model Architecture | Model Identifier | Original Authors & Organization | License | Description & Usage |
| :--- | :--- | :--- | :--- | :--- |
| **CLIP (`ViT-Base-Patch32`)** | `openai/clip-vit-base-patch32` | **OpenAI** (Radford et al., 2021) | MIT License | Multimodal zero-shot vision-language pre-trained model. Used locally for extracting `512-dimensional` continuous semantic embeddings for both raster images and natural language query strings. |

* **Original Research Paper:** *Learning Transferable Visual Models From Natural Language Supervision* (Radford, Kim, Hallacy, et al., ICML 2021). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
* **Hugging Face Model Hub Repository:** [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)

---

## 2. Core Open-Source Libraries & Frameworks

| Library / Framework | Version / Branch | Maintainers & Community | License | Role in System Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (`torch`, `torchvision`)**| `2.x` | Meta AI & PyTorch Foundation | BSD-3-Clause | Core neural network computation engine, tensor serialization (`.photo_index.pt`), and hardware acceleration (`CUDA / MPS / CPU`). |
| **Hugging Face Transformers** | `4.x` | Hugging Face Inc. | Apache 2.0 | Loading and executing `CLIPModel` and `CLIPProcessor` pipelines, BPE tokenization, and vision preprocessing. |
| **Streamlit (`streamlit`)** | `1.x` | Snowflake Inc. / Streamlit | Apache 2.0 | Frontend presentation layer, interactive responsive card grids, strictness threshold sliders, and stateful tab management. |
| **Pillow (`PIL`)** | `10.x` | Alex Clark & Contributors | HPND License | Raster image decoding (`JPG, PNG, WEBP, BMP, TIFF`) and RGB color space standardization before tensor conversion. |
| **Python Standard Library** | `3.10+` | Python Software Foundation | PSF License | `concurrent.futures` multi-threaded worker pools, `os.walk` directory traversal, and `time/psutil` profiling. |

---

## 3. Benchmark & Sample Datasets

During setup (`download_samples.py`), the system retrieves high-resolution sample photographs from public domain / free license repositories to benchmark retrieval precision and verify out-of-the-box functionality:

* **Unsplash Dataset Samples (`images.unsplash.com`):**
  - **Dog in Park:** Photo by [Unsplash Community] (`photo-1543466835-00a7907e9de1`), licensed under the **Unsplash License** (Free for commercial and non-commercial use, no attribution required).
  - **Beach Sunset:** Photo by [Unsplash Community] (`photo-1507525428034-b723cf961d3e`), licensed under the **Unsplash License**.
  - **Red Sports Car:** Photo by [Unsplash Community] (`photo-1503376780353-7e6692767b70`), licensed under the **Unsplash License**.
  - **Cat Sleeping:** Photo by [Unsplash Community] (`photo-1514888286974-6c03e2ca1dba`), licensed under the **Unsplash License**.
  - **Coffee Cup on Desk:** Photo by [Unsplash Community] (`photo-1495474472287-4d71bcdd2085`), licensed under the **Unsplash License**.

---

## 4. Pre-Existing Work & Inspiration

* **Local AI Search Assistants:** Inspired by privacy-centric local desktop tools (such as Apple Photos on-device neural search and open-source personal knowledge assistants) that prioritize user sovereignty over cloud synchronization.
* **Streamlit Custom Styling Patterns:** UI responsiveness and clean card styling draw inspiration from modern glassmorphism design systems and community Streamlit CSS snippets (`#MainMenu` suppression and hover micro-animations).

---

## 5. License Statement

This hackathon project is open-source and released under the **MIT License**. All bundled code, custom workflows, multi-threaded pipelines, and UI designs created for **OSDHack 2026** can be freely used, modified, and distributed by the open-source community.
