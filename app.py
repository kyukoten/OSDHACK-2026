import os
import torch
import streamlit as st
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures

# 1. Page Configuration & Custom CSS (Hackathon Polish)
st.set_page_config(page_title="Privacy-First Photo Search", page_icon="📷", layout="wide")

# Inject Custom CSS to make it look like a native app (hides Streamlit footers/menus and styles images)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stImage > img {
            border-radius: 12px;
            box-shadow: 0 4px 12px 0 rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .stImage > img:hover {
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

st.title("📷 Privacy-First Smart Photo Search (`Multi-Scale Vision Engine`)")
st.markdown("*Search your local photos using natural language with multi-crop feature fusion & prompt ensembling. 100% private & offline.*")

# 2. Load Model
@st.cache_resource(show_spinner="Loading AI Vision Engine into Memory...")
def load_clip_model():
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

try:
    model, processor = load_clip_model()
except Exception as e:
    st.error(f"Failed to load local model: {e}")
    st.stop()

# 3. Helpers and AI Logic
def get_image_paths(directory):
    if not os.path.exists(directory):
        return []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
    all_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_paths.append(os.path.join(root, file))
    return all_paths

def generate_or_load_embeddings(paths, directory):
    index_file = os.path.join(directory, ".photo_index.pt")
    
    if os.path.exists(index_file):
        try:
            stored_data = torch.load(index_file)
            if set(stored_data["paths"]) == set(paths):
                return stored_data["embeddings"], stored_data["paths"], []
        except Exception:
            pass

    embeddings, valid_paths, error_messages = [], [], []
    progress_bar = st.progress(0, text="Analyzing new images with multi-crop vision fusion...")
    
    def process_single_image(path):
        try:
            # 1. Load image and automatically correct EXIF orientation (rotation issues from phone cameras)
            raw_img = Image.open(path)
            raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
            
            # 2. Multi-Crop Feature Fusion: Extract both full scale and a center zoom crop
            # This allows CLIP to capture both global scene context and small/off-center object details
            w, h = raw_img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            center_crop = raw_img.crop((left, top, left + crop_size, top + crop_size))
            
            inputs_full = processor(images=raw_img, return_tensors="pt", padding=True)
            inputs_crop = processor(images=center_crop, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                feat_full = model.visual_projection(model.vision_model(**inputs_full).pooler_output)
                feat_crop = model.visual_projection(model.vision_model(**inputs_crop).pooler_output)
                
                # Fuse features (70% full scene + 30% center object focus) and L2 normalize
                fused_features = 0.7 * feat_full + 0.3 * feat_crop
                fused_features /= fused_features.norm(dim=-1, keepdim=True)
                
            return (path, fused_features, None)
        except Exception as img_err:
            return (path, None, f"Error: {str(img_err)}")

    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_image, path): path for path in paths}
        for future in concurrent.futures.as_completed(futures):
            path, feature, error = future.result()
            if feature is not None:
                embeddings.append(feature)
                valid_paths.append(path)
            if error:
                error_messages.append(error)
            processed_count += 1
            progress_bar.progress(processed_count / len(paths), text=f"Processing {processed_count}/{len(paths)} images...")
            
    progress_bar.empty()
            
    if embeddings:
        db_embeddings = torch.cat(embeddings, dim=0)
        torch.save({"embeddings": db_embeddings, "paths": valid_paths}, index_file)
        return db_embeddings, valid_paths, error_messages
        
    return None, [], error_messages

# 4. App Layout (Tabs Interface)
tab_search, tab_settings = st.tabs(["🔍 Search Gallery", "⚙️ Engine & Database"])

# --- TAB 2: SETTINGS & DATABASE INIT ---
with tab_settings:
    st.header("Database Configuration")
    target_dir = st.text_input("Local Image Directory Path", value="./sample_images")
    image_paths = get_image_paths(target_dir)
    
    if not image_paths:
        st.warning(f"No valid images found in `{target_dir}`. Please add some photos.")
        db_embeddings, indexed_paths = None, []
    else:
        # Generate stats
        col1, col2 = st.columns(2)
        col1.metric("Images Found", len(image_paths))
        col2.metric("Index Status", "Active" if os.path.exists(os.path.join(target_dir, ".photo_index.pt")) else "Requires Build")
        
        db_embeddings, indexed_paths, processing_errors = generate_or_load_embeddings(image_paths, target_dir)

        if processing_errors:
            with st.expander("⚠️ View Image Processing Errors"):
                for err in processing_errors[:5]:
                    st.write(err)

# --- TAB 1: SEARCH INTERFACE ---
with tab_search:
    if db_embeddings is None:
        st.info("👈 Please configure a valid image directory in the 'Engine & Database' tab first.")
    else:
        # Clean search bar layout with sorting options
        col_search, col_slider, col_sort = st.columns([2.5, 1, 1])
        with col_search:
            query = st.text_input("What are you looking for?", placeholder="e.g., a dog playing, a red car, sunset...", label_visibility="collapsed")
        with col_slider:
            threshold = st.slider("Strictness Threshold", min_value=15, max_value=60, value=25, help="Filter out matches below this confidence level.")
        with col_sort:
            sort_by = st.selectbox("Sort Results By", ["🔥 Highest Match", "📁 File Name (A-Z)"], label_visibility="collapsed")

        if query:
            with st.spinner("Executing Prompt Ensembling & Multi-Scale Scan..."):
                # 3. Prompt Ensembling (Zero-Shot Prompt Expansion)
                # Instead of embedding just the raw query, we ensemble natural photographic templates to boost zero-shot accuracy
                prompt_templates = [
                    f"a photo of {query}",
                    f"a picture showing {query}",
                    f"a close-up photograph of {query}",
                    query
                ]
                text_inputs = processor(text=prompt_templates, return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_outputs = model.text_model(**text_inputs)
                    text_features_list = model.text_projection(text_outputs.pooler_output)
                    # Average the prompt embeddings and L2 normalize
                    text_features = torch.mean(text_features_list, dim=0, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity_scores = (db_embeddings @ text_features.T).squeeze(dim=-1)
                
                top_k = min(18, len(indexed_paths))
                top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
                
            st.markdown("---")
            
            # Filter and sort results
            results = []
            for score, index in zip(top_scores, top_indices):
                match_percentage = int(score.item() * 100)
                if match_percentage >= threshold:
                    img_path = indexed_paths[index.item()]
                    results.append({"path": img_path, "score": match_percentage, "index": index.item()})
            
            if sort_by == "📁 File Name (A-Z)":
                results.sort(key=lambda x: os.path.basename(x["path"]).lower())
            
            # Interactive Grid layout
            cols = st.columns(3)
            displayed_count = 0
            
            for item in results:
                col_target = cols[displayed_count % 3]
                img_path = item["path"]
                match_percentage = item["score"]
                
                # Extract image dimensions cleanly for metadata badge
                try:
                    with Image.open(img_path) as im:
                        img_dim = f"{im.width}x{im.height}"
                except Exception:
                    img_dim = "Unknown"
                
                with col_target:
                    st.image(img_path, use_container_width=True)
                    # Styled caption with match badge and resolution
                    st.markdown(f"🔥 **Match: `{match_percentage}%`**  |  📐 `{img_dim}`")
                    st.caption(f"📂 `{os.path.basename(img_path)}`")
                
                displayed_count += 1
                    
            if displayed_count == 0:
                st.warning(f"No matches found for '{query}' above {threshold}% strictness. Try lowering the strictness slider or adjusting your prompt.")