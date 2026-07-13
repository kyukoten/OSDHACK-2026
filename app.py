import os
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Page Configuration
st.set_page_config(page_title="Privacy-First Photo Search", layout="wide")
st.title("📷 Privacy-First Smart Photo Search")
st.write("Search your local photos using natural language. Zero cloud dependencies. 100% private.")

# 2. Load Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_clip_model():
    # Utilizing a lightweight, fast-inference CLIP variant ideal for on-device processing
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

try:
    model, processor = load_clip_model()
except Exception as e:
    st.error(f"Failed to load local model: {e}")
    st.stop()

# 3. Sidebar for Settings
st.sidebar.header("📁 Target Configuration")
target_dir = st.sidebar.text_input("Local Image Directory Path", value="./sample_images")

# Helper to load valid image paths
def get_image_paths(directory):
    if not os.path.exists(directory):
        return []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

image_paths = get_image_paths(target_dir)

# 4. Core AI Logic: Indexing & Feature Extraction
@st.cache_data(show_spinner="Analyzing and indexing local images...")
def generate_embeddings(paths):
    embeddings = []
    valid_paths = []
    error_messages = []
    
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                # 1. Run image through the vision encoder explicitly
                vision_outputs = model.vision_model(**inputs)
                # 2. Extract the pooled tensor and project it to the multimodal space
                image_features = model.visual_projection(vision_outputs.pooler_output)
                # 3. Normalize the raw tensor
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features)
            valid_paths.append(path)
        except Exception as img_err:
            error_messages.append(f"Error processing `{os.path.basename(path)}`: {str(img_err)}")
            continue
            
    return embeddings, valid_paths, error_messages

# 5. Execution Pipeline
if not image_paths:
    st.warning(f"No valid images found in `{target_dir}`. Please create the folder and add some JPG/PNG files.")
else:
    st.sidebar.success(f"Found {len(image_paths)} images to index.")
    
    # Run local indexing
    embeddings_list, indexed_paths, processing_errors = generate_embeddings(image_paths)

    # Show debugging info if images failed to process
    if processing_errors:
        with st.expander("⚠️ View Image Processing Errors"):
            for err in processing_errors[:5]: # Show first 5 errors
                st.write(err)
            if len(processing_errors) > 5:
                st.write(f"...and {len(processing_errors) - 5} more errors.")

    if not embeddings_list:
        st.error("❌ Critical: Failed to generate embeddings for any of your images. See the error dropdown above.")
        db_embeddings = None
    else:
        db_embeddings = torch.cat(embeddings_list, dim=0)

    # 6. Search Interface
    query = st.text_input("Search Prompt", placeholder="e.g., a dog playing in the park, a sunset by the beach, group photo")

    if query and db_embeddings is not None:
        with st.spinner("Searching local index..."):
            text_inputs = processor(text=[query], return_tensors="pt", padding=True)
            
            with torch.no_grad():
                # 1. Run text through the text encoder explicitly
                text_outputs = model.text_model(**text_inputs)
                # 2. Extract and project the text tensor
                text_features = model.text_projection(text_outputs.pooler_output)
                # 3. Normalize the raw tensor
                text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity match scores matrix locally
            similarity_scores = (db_embeddings @ text_features.T).squeeze(dim=-1)
            
            # Sort items by highest match ranking values
            top_k = min(12, len(indexed_paths)) # Increased from 6 to 12
            top_scores, top_indices = torch.topk(similarity_scores, k=top_k)
            
        # Display Grid Results
        # Display Grid Results
        st.write(f"### Top Matches for '{query}':")
        
        # Hackathon Feature: A slider to filter out low-confidence "junk" matches
        threshold = st.slider("Match Strictness Threshold (%)", min_value=10, max_value=50, value=25)
        
        cols = st.columns(3)
        displayed_count = 0
        
        for idx, (score, index) in enumerate(zip(top_scores, top_indices)):
            match_percentage = int(score.item() * 100)
            
            # Only show the image if it meets our strictness threshold
            if match_percentage >= threshold:
                col_target = cols[displayed_count % 3]
                img_path = indexed_paths[index.item()]
                
                with col_target:
                    # FIX: Updated to use_container_width to remove the yellow warnings
                    st.image(img_path, use_container_width=True)
                    st.caption(f"**Match:** {match_percentage}% | `{os.path.basename(img_path)}`")
                
                displayed_count += 1
                
        if displayed_count == 0:
            st.info(f"No images matched your query with a confidence score above {threshold}%. Try a lower threshold or a different prompt.")