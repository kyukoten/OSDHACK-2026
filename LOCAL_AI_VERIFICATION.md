# 🔒 Local AI Verification & Network Isolation Audit

This document formally verifies and outlines the local-first execution guarantees of **Privacy-First Smart Photo Search**, prepared for **OSDHack 2026**. It provides clear audit boundaries detailing what executes entirely on-device, where network requests occur during setup, and proof that **zero user data ever leaves the local hardware**.

---

## 1. On-Device Execution Matrix

The table below breaks down every functional module of the application and specifies where compute occurs:

| Module / Operation | Execution Environment | Network Access Required? | External Cloud Dependency | User Data Exposed? |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit Web UI (`app.py`)** | Local Desktop (`localhost:8501`) | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Directory File Scanning** | Local OS Filesystem (`os.walk`) | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Image Decoding (`PIL/RGB`)** | Multi-Core Local CPU | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Vision Transformer Forward Pass** | Local PyTorch Runtime (CPU/GPU) | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Text Tokenization & Encoding** | Local PyTorch Runtime (CPU/GPU) | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Vector Index Cache (`.photo_index.pt`)**| Local Hard Drive (`C:\...`) | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Cosine Similarity Matrix `@`** | Local PyTorch Tensor Math | ❌ **No** | None (`Offline`) | ❌ **No** |
| **Top-K Ranking & Strictness Filter** | Local PyTorch Math Engine | ❌ **No** | None (`Offline`) | ❌ **No** |

---

## 2. Internet Access & Setup Lifecycle

To ensure total transparency, the application's network interactions are strictly segregated into two distinct phases:

### Phase A: Initial Setup & One-Time Asset Caching (Internet Required)
During the very first execution (`pip install -r requirements.txt` and initial launch of `app.py`), the following outbound network requests occur:
1. **PyPI Package Index (`pypi.org`):** Downloading local binaries (`torch`, `transformers`, `pillow`, `streamlit`).
2. **Hugging Face Model Hub (`huggingface.co`):** Downloading the pre-trained `openai/clip-vit-base-patch32` weights and BPE vocabulary files (`~605 MB total`). These weights are cached locally inside the host OS cache directory (`~/.cache/huggingface/hub/`).
3. **Optional Sample Dataset Download (`download_samples.py`):** If the user executes the test script, 5 public domain test photos are downloaded from Unsplash (`images.unsplash.com`) into `./sample_images`.

### Phase B: Operational Mode (100% Air-Gapped / Offline)
Once Phase A completes, **no network requests are made by the application**.
* You can physically unplug your Ethernet cable, turn off Wi-Fi, or block Python via OS Firewall.
* Indexing new local photo directories, computing image feature vectors, saving/loading `.photo_index.pt`, and querying photos using complex natural language sentences will execute with **100% functionality and zero network latency**.

---

## 3. User Data Isolation Guarantee

### Does any user data leave the device?
**ABSOLUTELY NOT.**

1. **No Cloud Telemetry or Tracking:** The application does not integrate analytics scripts, error-reporting trackers (like Sentry or Bugsnag), or cloud logging APIs.
2. **No Remote API Calls for Inference:** Unlike cloud AI wrappers (e.g., OpenAI ChatGPT APIs, Google Cloud Vision, AWS Rekognition) where user photos and search prompts are serialized and transmitted over HTTPS to remote server farms, this application instantiates the neural network directly inside the host system's RAM/VRAM.
3. **Local Vector Security:** Mathematical representations (`512-dim` float vectors) generated from personal photos never touch an external network socket. They are stored strictly in the local working directory inside `.photo_index.pt`. Even if an external actor intercepted this local file, reconstructing original high-resolution personal photographs from `512-dimensional` L2-normalized embeddings is mathematically infeasible.

---

## 4. How Judges & Users Can Verify Isolation

We encourage hackathon judges and evaluators to independently audit and verify our air-gapped claim:

### Test Procedure (Air-Gapped Audit)
1. **Prepare Environment:** Clone the repo and run `pip install -r requirements.txt`.
2. **Warm Cache:** Run `streamlit run app.py` once while connected to the internet so Hugging Face caches the `ViT-B/32` weights locally.
3. **Sever Network:** Disconnect your Wi-Fi or disable the network adapter completely.
4. **Execute Offline Search:**
   - Launch `streamlit run app.py` again.
   - Go to **Engine & Database**, enter a local folder path of your private family/work photos (`C:\Users\YourName\Pictures`), and let it build the index.
   - Switch to **Search Gallery** and search for complex prompts (e.g., `"birthday cake"`, `"document with charts"`, `"group selfie at night"`).
   - **Result:** The system will successfully index and retrieve exact photos instantaneously without any internet connection.
