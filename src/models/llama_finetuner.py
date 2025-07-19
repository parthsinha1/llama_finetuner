import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import logging
from typing import Dict, List, Optional
import json

class LlamaFineTuner:
    """Fine-tune Llama 3.2-1B model using LoRA"""
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.2-1B",
                 use_quantization: bool = True,
                 use_lora: bool = True):
        
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # this initializes model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        self._setup_model_and_tokenizer()
    
    def _setup_model_and_tokenizer(self):
                
        self.logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # enable quantization if GPU + bitsandbytes are available
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        except ImportError:
            pass
        
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if self.use_lora:
            self._setup_lora()
        
        self.logger.info("Model and tokenizer loaded successfully!")
    
    def _setup_lora(self):
        
        lora_config = LoraConfig(
            r=16,  
            lora_alpha=32,  
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],  
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.model.print_trainable_parameters()
        
        self.logger.info("LoRA configuration applied!")
    
    def prepare_dataset(self, data_path: str, max_length: int = 512):
        
        self.logger.info(f"Loading dataset from: {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Currently only JSON format is supported")
        
        formatted_data = []
        for item in data:
            if item.get('input', '').strip():
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction: {item['instruction']}\nInput: {item['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}<|eot_id|>"
            else:
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction: {item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['output']}<|eot_id|>"
            
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        self.train_dataset = split_dataset["train"]
        self.eval_dataset = split_dataset["test"]
        
        self.logger.info(f"Dataset prepared: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval samples")
        
        return self.train_dataset, self.eval_dataset
    
    def train(self, 
              output_dir: str = "./results/llama-finetuned",
              num_epochs: int = 3,
              batch_size: int = 2,
              learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4):
        
        self.logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        self.trainer.train()
        
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Training completed! Model saved to: {output_dir}")
    
    def generate_response(self, 
                         instruction: str, 
                         input_text: str = "",
                         max_length: int = 200,
                         temperature: float = 0.7):
        
        if input_text.strip():
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction: {instruction}\nInput: {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nInstruction: {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response = full_response[len(prompt):]
        
        return response.strip()