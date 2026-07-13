# Privacy-First Smart Photo Search (On-Device AI)

A fully open-source, offline-first native desktop assistant built for **OSDHack 2026**. This application allows users to search through localized, personal image directories using unstructured semantic text queries (e.g., "sunset at the beach", "dog playing with a ball") without passing a single pixel or query string to external third-party cloud engines.

## 🚀 Core Features
- **100% On-Device Processing:** Runs completely local. Disconnect your internet completely and the engine functions fully.
- **Semantic Text-to-Image Search:** Powered by local Contrastive Language-Image Pretraining (CLIP) architectures.
- **Instant In-Memory Vector Indexing:** Extracts multidimensional text and image vector embeddings locally for rapid analytical sorting.

## 🛠️ Technical Architecture
The application leverages the **OpenAI CLIP (ViT-Base-Patch32)** multimodal model architecture deployed on local consumer hardware via the Hugging Face ecosystem. 

1. **Feature Extraction:** Images are converted into raw multi-dimensional tensors, passing through a localized Vision Transformer network to derive clean image descriptors.
2. **Vector Space Mapping:** Natural language search terms are mapped into the exact same vector space mathematical coordinates as the images.
3. **Similarity Assessment:** Local dot-product cross-multiplication matrix loops determine exact rank matching without utilizing proprietary cloud database nodes.

## 📦 Local Installation & Setup

1. **Clone the Repository:**
```bash
git clone [https://github.com/kyukoten/privacy-photo-search.git](https://github.com/kyukoten/privacy-photo-search.git)
cd privacy-photo-search