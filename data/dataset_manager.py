import os
import psutil
from huggingface_hub import HfApi
from datasets import load_dataset
import pandas as pd

# Global Memory Registry for files uploaded via Local Data Sources tab
GLOBAL_WORKSPACE_DATA = {}

class DatasetManager:
    def __init__(self, cache_dir="storage/datasets_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.api = HfApi()

        # Hardcoded specific dataset mappings for our enterprise categories
        # as HuggingFace search can be noisy
        self.category_map = {
            "Local Workspace Data": [], # Dynamically pulled from GLOBAL_WORKSPACE_DATA
            "AML & Financial Fraud": ['dvilasuero/banking_with_vectors', 'zeroshot/twitter-financial-news-sentiment'],
            "Customer Profiles & KYC": ['bitext/Bitext-customer-support-llm-chatbot-training-dataset'],
            "Synthetic Transactions": ['gretelai/synthetic_pii_finance_multilingual']
        }

    def fetch_datasets_by_category(self, category):
        """Return a list of dataset IDs for a given category."""
        if category == "Custom Hugging Face Dataset":
            return []
            
        if category == "Local Workspace Data":
            return list(GLOBAL_WORKSPACE_DATA.keys())
            
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

    def load_dataset_records(self, dataset_id, limit_str):
        """
        Load dataset records lazily based on the selected limit.
        limit_str can be '100', '1000', '10000', '100000', '1000000', or 'ALL'
        """
        try:
            limit = None if limit_str == "ALL" else int(limit_str)
            
            # If dataset is locally uploaded, serve it from RAM
            if dataset_id in GLOBAL_WORKSPACE_DATA:
                df = GLOBAL_WORKSPACE_DATA[dataset_id]
                if limit:
                    return df.head(limit)
                return df

            # We use streaming=True to avoid downloading huge files if the user only wants 100 rows
            # Load dataset lazily without hardcoded split
            ds_dict = load_dataset(dataset_id, streaming=True, cache_dir=self.cache_dir)
            split_name = list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            
            if limit:
                ds = ds.take(limit)
                
            # Convert stream to pandas
            records = list(ds)
            df = pd.DataFrame(records)
            return df
            
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
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        return "Cache cleared."
