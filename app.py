import os
import torch
import streamlit as st
from PIL import Image
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

st.title("📷 Privacy-First Smart Photo Search")
st.markdown("*Search your local photos using natural language. Zero cloud dependencies. 100% private.*")

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
    progress_bar = st.progress(0, text="Analyzing new images with multi-threading...")
    
    def process_single_image(path):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                vision_outputs = model.vision_model(**inputs)
                image_features = model.visual_projection(vision_outputs.pooler_output)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return (path, image_features, None)
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
        # Clean search bar layout
        col_search, col_slider = st.columns([3, 1])
        with col_search:
            query = st.text_input("What are you looking for?", placeholder="e.g., a dog playing, a red car, sunset...", label_visibility="collapsed")
        with col_slider:
            threshold = st.slider("Strictness", min_value=15, max_value=50, value=25, help="Increase to hide lower-confidence matches.")

        if query:
            with st.spinner("Scanning vector space..."):
                text_inputs = processor(text=[query], return_tensors="pt", padding=True)
                with torch.no_grad():
                    text_outputs = model.text_model(**text_inputs)
                    text_features = model.text_projection(text_outputs.pooler_output)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity_scores = (db_embeddings @ text_features.T).squeeze(dim=-1)
                
                top_k = min(12, len(indexed_paths))
                top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
                
            st.markdown("---")
            
            # Interactive Grid layout
            cols = st.columns(3)
            displayed_count = 0
            
            for idx, (score, index) in enumerate(zip(top_scores, top_indices)):
                match_percentage = int(score.item() * 100)
                
                if match_percentage >= threshold:
                    col_target = cols[displayed_count % 3]
                    img_path = indexed_paths[index.item()]
                    
                    with col_target:
                        st.image(img_path, use_container_width=True)
                        # Styled caption
                        st.markdown(f"**Match:** `{match_percentage}%` | 📂 `{os.path.basename(img_path)}`")
                    
                    displayed_count += 1
                    
            if displayed_count == 0:
                st.warning(f"No high-confidence matches found for '{query}'. Try lowering the strictness slider or changing the prompt.")