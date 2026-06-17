import os
import psutil
from huggingface_hub import HfApi
from datasets import load_dataset
import pandas as pd
import multiprocessing

def _download_worker(dataset_id, cache_dir):
    """
    Runs in a completely isolated OS process to bypass the Python GIL.
    Forces Hugging Face to download the dataset natively via Rust/Xet to the hard drive cache.
    """
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    try:
        from datasets import load_dataset
        
        if dataset_id.startswith("http://") or dataset_id.startswith("https://"):
            if dataset_id.endswith(".csv"):
                load_dataset("csv", data_files=dataset_id, cache_dir=cache_dir)
            elif dataset_id.endswith(".json"):
                load_dataset("json", data_files=dataset_id, cache_dir=cache_dir)
        else:
            load_dataset(dataset_id, cache_dir=cache_dir)
    except Exception as e:
        safe_id = str(dataset_id).replace("/", "_").replace("\\", "_")
        with open(os.path.join(cache_dir, f"{safe_id}_error.txt"), "w") as f:
            f.write(str(e))

# Multi-Tenant Memory Registry: {username: {dataset_name: dataframe}}
USER_WORKSPACE_DATA = {}

def get_user_workspace(username):
    if not username:
        username = "GUEST"
    if username not in USER_WORKSPACE_DATA:
        USER_WORKSPACE_DATA[username] = {}
    return USER_WORKSPACE_DATA[username]

class DatasetManager:
    def __init__(self, cache_dir="storage/datasets_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api = HfApi()
        import uuid
        self.background_tasks = {}

        # Hardcoded specific dataset mappings for our enterprise categories
        # as HuggingFace search can be noisy
        self.category_map = {
            "Local Workspace Data": [], # Dynamically pulled via API
            "AML & Financial Fraud": ['dvilasuero/banking_with_vectors', 'zeroshot/twitter-financial-news-sentiment'],
            "Customer Profiles & KYC": ['bitext/Bitext-customer-support-llm-chatbot-training-dataset'],
            "Synthetic Transactions": ['gretelai/synthetic_pii_finance_multilingual']
        }

    def fetch_datasets_by_category(self, category):
        """Return a list of dataset IDs for a given category."""
        if category == "Custom Hugging Face Dataset":
            return []
            
        if category == "Local Workspace Data":
            # Note: This is an internal check now, UI should call get_user_workspace
            return []
            
        # For a real application, we might search HF using self.api.list_datasets()
        # but for reliability we use our curated list.
        return self.category_map.get(category, [])

    def get_dataset_metadata(self, dataset_id):
        """Fetch metadata for a dataset using huggingface_hub API."""
        try:
            info = self.api.dataset_info(dataset_id)
            return {
                "id": info.id,
                "description": info.description[:100] + "..." if info.description else "No description available",
                "downloads": getattr(info, 'downloads', 0),
                "last_modified": getattr(info, 'lastModified', 'Unknown')
            }
        except Exception as e:
            return {"error": str(e)}

    def start_async_download(self, dataset_id, limit_str, username="GUEST"):
        import uuid
        task_id = str(uuid.uuid4())
        
        # Pre-process URLs before handing off to multiprocessing
        if dataset_id.startswith("http://") or dataset_id.startswith("https://"):
            if "huggingface.co/datasets/" in dataset_id:
                dataset_id = dataset_id.split("huggingface.co/datasets/")[-1].strip("/")
                
        if dataset_id == "conceptual_captions":
            dataset_id = "google-research-datasets/conceptual_captions"
        elif dataset_id == "sbu_captions":
            dataset_id = "vicenteor/sbu_captions"
            
        p = multiprocessing.Process(target=_download_worker, args=(dataset_id, self.cache_dir))
        p.daemon = True
        p.start()
        
        self.background_tasks[task_id] = {
            "status": "DOWNLOADING",
            "dataset_id": dataset_id,
            "limit_str": limit_str,
            "username": username,
            "process": p
        }
        
        return task_id

    def _load_dataset_records_sync(self, dataset_id, limit_str, username="GUEST"):
        """
        Load dataset records lazily based on the selected limit.
        limit_str can be '100', '1000', '10000', '100000', '1000000', or 'ALL'
        """
        try:
            limit = None if limit_str == "ALL" else int(limit_str)
            
            # If dataset is locally uploaded, serve it from RAM
            user_workspace = get_user_workspace(username)
            if dataset_id in user_workspace:
                df = user_workspace[dataset_id]
                if limit:
                    return df.head(limit)
                return df
            
            # Route all data through Hugging Face Apache Arrow pipeline to enforce Zero-RAM
            if dataset_id.startswith("http://") or dataset_id.startswith("https://"):
                if dataset_id.endswith(".csv"):
                    ds_dict = load_dataset("csv", data_files=dataset_id, cache_dir=self.cache_dir)
                elif dataset_id.endswith(".json"):
                    ds_dict = load_dataset("json", data_files=dataset_id, cache_dir=self.cache_dir)
                else:
                    return pd.DataFrame([{"Error": "Unsupported URL format. Only .csv, .json, or Hugging Face repos allowed."}])
            else:
                ds_dict = load_dataset(dataset_id, cache_dir=self.cache_dir)
            split_name = list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
                
            # ZERO RAM ARCHITECTURE: Return the memory mapped Arrow pointer directly!
            return ds
            
        except Exception as e:
            return pd.DataFrame([{"Error": f"Failed to load dataset {dataset_id}: {str(e)}"}])

    def estimate_memory_and_time(self, limit_str):
        """Estimate processing time and memory usage based on record count."""
        if limit_str == "ALL":
            return "High", "Unknown"
            
        limit = int(limit_str)
        mem_mb = (limit * 1.5) / 1024 # Very rough approximation: ~1.5KB per row
        
        if limit <= 1000:
            time_est = "< 1 second"
        elif limit <= 100000:
            time_est = "~ 5 seconds"
        else:
            time_est = "~ 30+ seconds"
            
        return f"{mem_mb:.2f} MB", time_est

    def get_cache_info(self):
        """Return cache size and location."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return {
            "size_mb": total_size / (1024 * 1024),
            "location": os.path.abspath(self.cache_dir)
        }

    def clear_cache(self):
        import shutil
        import gc
        
        # 1. Clear Active RAM Sessions / Memory Mapped Pointers FIRST
        global USER_WORKSPACE_DATA
        USER_WORKSPACE_DATA.clear()
        
        # 2. Force Garbage Collection to close Windows file locks on memory-mapped Arrow files
        gc.collect()
        
        # 3. Clear Local Hard Drive Cache
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 4. Clear Global Hugging Face Download Cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        shutil.rmtree(hf_cache, ignore_errors=True)
        os.makedirs(hf_cache, exist_ok=True)
        
        return "Cache, Pointers, and Global HF Downloads cleared."

    def get_cached_datasets(self):
        """Scan the local hard drive for downloaded dataset folders."""
        if not os.path.exists(self.cache_dir):
            return ["No Datasets Cached"]
        
        folders = []
        for d in os.listdir(self.cache_dir):
            if os.path.isdir(os.path.join(self.cache_dir, d)):
                # Huggingface caches repos with '___' instead of '/'
                if "___" in d:
                    folders.append(d.replace("___", "/"))
                else:
                    folders.append(d)
                    
        if not folders:
            return ["No Datasets Cached"]
        return folders

    def delete_specific_dataset(self, dataset_folder):
        if dataset_folder == "No Datasets Cached" or not dataset_folder:
            return "No valid dataset selected."
            
        import shutil
        import gc
        
        target_path = os.path.join(self.cache_dir, dataset_folder)
        if os.path.exists(target_path):
            # Clear Active RAM Sessions / Memory Mapped Pointers FIRST
            global USER_WORKSPACE_DATA
            USER_WORKSPACE_DATA.clear()
            
            # Force Garbage Collection to close Windows file locks
            gc.collect()
            
            try:
                shutil.rmtree(target_path, ignore_errors=False)
                
                # AGGRESSIVELY WIPE GLOBAL HF DOWNLOAD CACHE
                # Hugging Face hides gigabytes of raw .parquet internet downloads here
                # Even though the dataset is extracted into storage/datasets_cache!
                hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
                if os.path.exists(hf_cache):
                    import stat
                    # Force delete readonly files
                    def remove_readonly(func, path, excinfo):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    shutil.rmtree(hf_cache, onerror=remove_readonly)
                    os.makedirs(hf_cache, exist_ok=True)
                    
                return f"Successfully deleted {dataset_folder} from cache."
            except Exception as e:
                return f"❌ Failed to delete (File Locked): Close the dataset in the UI or restart the server. ({str(e)})"
            
        return f"Dataset folder {dataset_folder} not found."
