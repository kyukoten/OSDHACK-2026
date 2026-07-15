import os
import sys
import time
import torch
import psutil
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def main():
    print("="*60)
    print("[*] PRIVACY-FIRST PHOTO SEARCH: TECHNICAL BENCHMARK & EVALUATION")
    print("="*60)
    
    start_mem = get_memory_usage()
    print(f"Base Process Memory: {format_bytes(start_mem)}")
    
    # 1. Benchmark Model Loading
    print("\n[1/5] Loading CLIP Model (openai/clip-vit-base-patch32)...")
    t0 = time.time()
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    load_time = time.time() - t0
    
    model.eval()
    post_load_mem = get_memory_usage()
    model_mem = post_load_mem - start_mem
    
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_model.parameters())
    text_params = sum(p.numel() for p in model.text_model.parameters())
    
    print(f"✅ Model Loaded in {load_time:.2f}s")
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Vision Encoder:   {vision_params:,} ({vision_params/1e6:.1f}M)")
    print(f"   Text Encoder:     {text_params:,} ({text_params/1e6:.1f}M)")
    print(f"   Memory Delta:     {format_bytes(model_mem)}")
    print(f"   Peak RAM Usage:   {format_bytes(post_load_mem)}")
    
    # 2. Benchmark Image Feature Extraction
    sample_dir = "./sample_images"
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_paths = []
    if os.path.exists(sample_dir):
        for root, dirs, files in os.walk(sample_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(root, file))
                    
    print(f"\n[2/5] Benchmarking Image Embedding Extraction ({len(image_paths)} images)...")
    if not image_paths:
        print("❌ No images found in ./sample_images. Please run download_samples.py first.")
        return
        
    image_embeddings = []
    extraction_times = []
    for path in image_paths:  # Benchmark all discovered images
        try:
            img = Image.open(path).convert("RGB")
            t_start = time.perf_counter()
            inputs = processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                vision_out = model.vision_model(**inputs)
                features = model.visual_projection(vision_out.pooler_output)
                features /= features.norm(dim=-1, keepdim=True)
            t_end = time.perf_counter()
            extraction_times.append((t_end - t_start) * 1000) # ms
            image_embeddings.append(features)
        except Exception as e:
            print(f"   Failed {path}: {e}")
            
    if extraction_times:
        avg_img_time = sum(extraction_times) / len(extraction_times)
        print(f"[+] Image Extraction Benchmarks:")
        print(f"   Average Latency per Image: {avg_img_time:.2f} ms")
        print(f"   Min Latency:               {min(extraction_times):.2f} ms")
        print(f"   Max Latency:               {max(extraction_times):.2f} ms")
        print(f"   Embedding Vector Dim:      {image_embeddings[0].shape[-1]}")
        
    # 3. Benchmark Text Query Feature Extraction
    print("\n[3/5] Benchmarking Text Query Embedding Extraction...")
    test_queries = [
        "a dog playing in the park",
        "sunset by the beach",
        "red sports car",
        "cat sleeping peacefully",
        "coffee cup on desk"
    ]
    query_times = []
    text_embeddings = []
    for q in test_queries:
        t_start = time.perf_counter()
        inputs = processor(text=[q], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_out = model.text_model(**inputs)
            features = model.text_projection(text_out.pooler_output)
            features /= features.norm(dim=-1, keepdim=True)
        t_end = time.perf_counter()
        query_times.append((t_end - t_start) * 1000) # ms
        text_embeddings.append(features)
        
    avg_query_time = sum(query_times) / len(query_times)
    print(f"[+] Text Query Benchmarks:")
    print(f"   Average Latency per Query: {avg_query_time:.2f} ms")
    print(f"   Min Latency:               {min(query_times):.2f} ms")
    print(f"   Max Latency:               {max(query_times):.2f} ms")
    
    # 4. Benchmark Cosine Similarity Search Latency (Matrix Product)
    print("\n[4/5] Benchmarking Vector Dot-Product Search Latency...")
    if image_embeddings and text_embeddings:
        db_tensor = torch.cat(image_embeddings, dim=0) # [N, 512]
        query_tensor = text_embeddings[0] # [1, 512]
        
        # Warmup
        _ = (db_tensor @ query_tensor.T).squeeze(dim=-1)
        
        search_times = []
        for _ in range(1000):
            t_s = time.perf_counter()
            scores = (db_tensor @ query_tensor.T).squeeze(dim=-1)
            top_scores, top_idx = torch.topk(scores, k=min(5, len(db_tensor)))
            t_e = time.perf_counter()
            search_times.append((t_e - t_s) * 1e6) # microseconds
            
        avg_search = sum(search_times) / len(search_times)
        print(f"[+] Search Latency (1000 runs averaged across {len(db_tensor)} items):")
        print(f"   Average Search & Top-K Time: {avg_search:.2f} us ({avg_search/1000:.4f} ms)")
        
    # 5. Qualitative/Quantitative Retrieval Evaluation on Known Ground Truth
    print("\n[5/5] Evaluating Top-1 Retrieval Accuracy on Sample Dataset...")
    # Map expected target filenames if present
    ground_truth = {
        "dog playing in the park": "dog_in_park.jpg",
        "sunset by the beach": "beach_sunset.jpg",
        "red sports car": "red_sports_car.jpg",
        "cat sleeping": "cat_sleeping.jpg",
        "coffee cup on desk": "coffee_cup_on_desk.jpg"
    }
    
    correct_top1 = 0
    evaluated = 0
    db_tensor = torch.cat(image_embeddings, dim=0)
    
    for query, expected_name in ground_truth.items():
        # Check if expected image exists in our image_paths
        matching_paths = [p for p in image_paths[:len(image_embeddings)] if expected_name in p]
        if not matching_paths:
            continue
        evaluated += 1
        
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_out = model.text_model(**inputs)
            features = model.text_projection(text_out.pooler_output)
            features /= features.norm(dim=-1, keepdim=True)
            
        scores = (db_tensor @ features.T).squeeze(dim=-1)
        top_score, top_idx = torch.topk(scores, k=1)
        best_match_path = image_paths[top_idx.item()]
        
        is_correct = expected_name in best_match_path
        if is_correct:
            correct_top1 += 1
        status = "[PASS]" if is_correct else "[FAIL]"
        print(f"   Query: '{query}' -> Best Match: {os.path.basename(best_match_path)} ({top_score.item()*100:.1f}%) {status}")
        
    if evaluated > 0:
        accuracy = (correct_top1 / evaluated) * 100
        print(f"\n[*] Top-1 Retrieval Accuracy on Benchmark Subset: {correct_top1}/{evaluated} ({accuracy:.1f}%)")
    else:
        print("\n[i] Ground truth sample images (dog_in_park.jpg, etc.) not in tested slice, skipping accuracy % computation.")
        
    print("\n" + "="*60)
    print(f"[*] Final Peak RAM Consumption: {format_bytes(get_memory_usage())}")
    print("="*60)

if __name__ == "__main__":
    main()
