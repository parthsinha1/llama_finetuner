#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.llama_finetuner import LlamaFineTuner
from src.data.prepare_alpaca import prepare_data

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2-1B")
    parser.add_argument("--data-size", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./results/llama-finetuned", help="Output directory")
    
    args = parser.parse_args()
    
    print(" Starting Llama 3.2-1B Fine-tuning")
    print(f" Data size: {args.data_size}")
    print(f" Epochs: {args.epochs}")
    print(f" Batch size: {args.batch_size}")
    
    print("\n Step 1: Preparing dataset...")
    data_path = prepare_data(sample_size=args.data_size)
    
    print("\n Step 2: Loading Llama model...")
    finetuner = LlamaFineTuner()
    
    print("\n Step 3: Processing dataset...")
    finetuner.prepare_dataset(data_path)
    
    print("\n Step 4: Starting training...")
    finetuner.train(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n Step 5: Testing fine-tuned model...")
    test_instructions = [
        "Explain machine learning in simple terms.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the benefits of regular exercise?"
    ]
    
    for instruction in test_instructions:
        print(f"\n Instruction: {instruction}")
        response = finetuner.generate_response(instruction)
        print(f" Response: {response}")
    
    print(f"\n Training completed! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()