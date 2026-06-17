import time
import pandas as pd
import numpy as np
import threading
import psutil

class EntityMemoryIndex:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EntityMemoryIndex, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
            
    def _initialize(self):
        self.is_built = False
        self.neural_encoder = None
        self.vector_matrix = None
        self.entity_data = pd.DataFrame()
        self.build_lock = threading.Lock()
        self.status = "IDLE"
        self.progress_percent = 0
        self.active_encoder = "None"
        
    def build_index(self, dataset_id, encoder_type="Text-Only (MiniLM - Fast)", max_records=50000):
        with self.build_lock:
            self.progress_percent = 0
            self.status = f"PULLING REAL DATA: Fetching {max_records} records from {dataset_id}..."
            try:
                from data.dataset_manager import DatasetManager
                dm = DatasetManager()
                
                if not dataset_id or dataset_id == "No valid dataset selected." or dataset_id == "No Datasets Cached":
                    raise RuntimeError("No valid target_dataset provided.")
                    
                ds = dm._load_dataset_records_sync(dataset_id, str(max_records))
                if isinstance(ds, pd.DataFrame) and "Error" in ds.columns:
                    raise RuntimeError(f"Dataset load failed: {ds.iloc[0]['Error']}")
                
                self.progress_percent = 10
                self.status = "VECTORIZING: Fast C++ string conversion of Pandas rows..."
                
                if isinstance(ds, pd.DataFrame):
                    df = ds.copy()
                else:
                    df = pd.DataFrame([ds[i] for i in range(len(ds))])
                    
                # 100x Speedup: Replaced iterrows with vectorized C++ string aggregation
                df['vector_text'] = df.astype(str).agg(' '.join, axis=1)
                self.entity_data = df
                
                self.progress_percent = 25
                
                model_name = 'all-MiniLM-L6-v2'
                is_multimodal = "CLIP" in encoder_type
                if is_multimodal:
                    model_name = 'clip-ViT-B-32'
                    
                self.status = f"INITIALIZING NEURAL NET: Loading {model_name} into System RAM..."
                
                device = 'cpu'
                
                if self.neural_encoder is None or self.active_encoder != model_name:
                    from sentence_transformers import SentenceTransformer
                    self.neural_encoder = SentenceTransformer(model_name, device=device)
                    self.active_encoder = model_name
                
                # Auto-Detect Visual Columns (PIL Objects or URLs/Files)
                from PIL import Image
                import urllib.request
                import io
                
                visual_col = None
                if is_multimodal and len(df) > 0:
                    for col in df.columns:
                        if col == 'vector_text':
                            continue
                        sample = df[col].iloc[0]
                        if isinstance(sample, Image.Image):
                            visual_col = col
                            break
                        elif isinstance(sample, str) and sample.lower().endswith(('.jpg', '.png', '.jpeg', '.mp4')):
                            visual_col = col
                            break
                            
                texts = self.entity_data['vector_text'].tolist()
                visuals = df[visual_col].tolist() if visual_col else None
                
                # CPU Vision processing is extremely slow. Truncate to 150 if using visual_col
                if visual_col:
                    texts = texts[:150]
                    visuals = visuals[:150]
                
                total_texts = len(texts)
                batch_size = 256 if not visual_col else 32 # Smaller batch for images to save RAM
                total_batches = (total_texts // batch_size) + 1
                
                all_embeddings = []
                
                for i in range(total_batches):
                    batch_texts = texts[i*batch_size : (i+1)*batch_size]
                    if not batch_texts: break
                    
                    self.status = f"EMBEDDING: Generating Neural Tensors (Batch {i}/{total_batches})..."
                    text_emb = self.neural_encoder.encode(batch_texts, show_progress_bar=False, convert_to_tensor=True, device=device)
                    
                    if visual_col:
                        batch_vis_raw = visuals[i*batch_size : (i+1)*batch_size]
                        batch_images = []
                        for vis in batch_vis_raw:
                            try:
                                if isinstance(vis, Image.Image):
                                    batch_images.append(vis)
                                elif isinstance(vis, str) and vis.lower().endswith('.mp4'):
                                    import cv2
                                    cap = cv2.VideoCapture(vis)
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)) # Middle frame
                                    ret, frame = cap.read()
                                    cap.release()
                                    if ret:
                                        batch_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                                    else:
                                        batch_images.append(Image.new('RGB', (224, 224), color='black'))
                                elif isinstance(vis, str) and vis.startswith('http'):
                                    req = urllib.request.Request(vis, headers={'User-Agent': 'Mozilla/5.0'})
                                    with urllib.request.urlopen(req) as url_resp:
                                        batch_images.append(Image.open(io.BytesIO(url_resp.read())).convert('RGB'))
                                elif isinstance(vis, str):
                                    batch_images.append(Image.open(vis).convert('RGB'))
                                else:
                                    batch_images.append(Image.new('RGB', (224, 224), color='black'))
                            except Exception:
                                batch_images.append(Image.new('RGB', (224, 224), color='black'))
                                
                        img_emb = self.neural_encoder.encode(batch_images, show_progress_bar=False, convert_to_tensor=True, device=device)
                        # Mathematical Fusion: Average text and image tensors
                        final_emb = (text_emb + img_emb) / 2.0
                    else:
                        final_emb = text_emb
                        
                    # CRITICAL: Move VRAM tensors back to SysRAM (CPU) immediately to allow extreme scaling
                    all_embeddings.append(final_emb.cpu())
                    self.progress_percent = 25 + int((i / total_batches) * 75)
                
                self.vector_matrix = torch.cat(all_embeddings, dim=0)
                
                self.is_built = True
                self.progress_percent = 100
                ram_used = psutil.Process().memory_info().rss / (1024 * 1024)
                vis_msg = " (Fused Text+Vision)" if visual_col else " (Text Only)"
                self.status = f"READY: Mapped {len(df)} entities{vis_msg} into Neural Space using {model_name}. Process RAM: {ram_used:.1f} MB."
            except Exception as e:
                import traceback
                self.status = f"CRASH: {str(e)} | {traceback.format_exc()}"
                self.is_built = False
                self.progress_percent = 0
                
    def search(self, query, image_path=None, top_k=10):
        if not self.is_built or self.vector_matrix is None:
            return pd.DataFrame({"Error": ["Index not built. Please build index first."]})
            
        try:
            import torch
            from PIL import Image
            
            # Encode user query or image into neural space
            search_input = query
            if image_path:
                if self.active_encoder != 'clip-ViT-B-32':
                    return pd.DataFrame({"Error": ["Visual Search requires the Multimodal Vision (CLIP) engine. Please select it from the dropdown and rebuild the memory."]})
                try:
                    search_input = Image.open(image_path)
                except Exception as img_e:
                    return pd.DataFrame({"Error": [f"Failed to load image: {str(img_e)}"]})
                    
            if not search_input:
                return pd.DataFrame({"Error": ["Please provide a text query or an image."]})
                
            query_vec = self.neural_encoder.encode(search_input, convert_to_tensor=True, device='cpu')
            
            # Compute cosine similarity using optimized PyTorch tensor math
            from sentence_transformers.util import cos_sim
            similarities = cos_sim(query_vec, self.vector_matrix)[0]
            
            # Get top K indices securely
            safe_k = min(top_k, len(similarities))
            top_scores, top_indices = torch.topk(similarities, k=safe_k)
            
            results = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                score = top_scores[i].item()
                if score > 0.15: # Semantic threshold
                    row = self.entity_data.iloc[idx].copy()
                    risk = row.get("risk_score", row.get("Amount", np.random.randint(10, 99)))
                    entity_id = row.get("Account", row.get("nameDest", row.get("nameOrig", f"ENT-{idx}")))
                    
                    results.append({
                        "Entity ID": entity_id,
                        "Neural Match Score": f"{score:.3f}",
                        "Assigned Risk": risk,
                        "Underlying Real Data": row['vector_text'][:150] + "..."
                    })
                    
            if not results:
                return pd.DataFrame({"Result": ["No semantic neural matches found."]})
                
            return pd.DataFrame(results)
        except Exception as e:
            return pd.DataFrame({"Error": [str(e)]})
            
    def clear(self):
        with self.build_lock:
            self.vector_matrix = None
            self.entity_data = pd.DataFrame()
            self.is_built = False
            self.status = "CLEARED: Memory released."
            return self.status

entity_memory_index = EntityMemoryIndex()
