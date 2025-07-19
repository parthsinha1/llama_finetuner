import json
import requests
import pandas as pd
from pathlib import Path
import logging

class AlpacaDataPreparer:
    """Download and prepare Alpaca dataset for fine-tuning"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def download_alpaca_data(self, output_path: str = "data/raw/alpaca_data.json"):
        
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        
        self.logger.info("Downloading Stanford Alpaca dataset...")
        
        # create directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # download data
        response = requests.get(url)
        response.raise_for_status()
        
        # save data
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        
        self.logger.info(f"Dataset downloaded to: {output_path}")
        return output_path
    
    def create_sample_dataset(self, 
                             input_path: str, 
                             output_path: str = "data/processed/alpaca_sample.json",
                             sample_size: int = 1000):
        """Create a smaller sample for faster training"""
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # sample data
        if len(data) > sample_size:
            import random
            random.seed(42)
            sampled_data = random.sample(data, sample_size)
        else:
            sampled_data = data
        
        # create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # save the sampled data
        with open(output_path, 'w') as f:
            json.dump(sampled_data, f, indent=2)
        
        self.logger.info(f"Created sample dataset with {len(sampled_data)} examples: {output_path}")
        return output_path

def prepare_data(sample_size: int = 1000):
    """Quick function to prepare data"""
    preparer = AlpacaDataPreparer()
    
    # download full dataset
    raw_path = preparer.download_alpaca_data()
    
    # create sample
    sample_path = preparer.create_sample_dataset(raw_path, sample_size=sample_size)
    
    return sample_path

if __name__ == "__main__":
    prepare_data()