---
base_model: meta-llama/Llama-3.2-1B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.2-1B
- lora
- transformers
---

# Model Card for Llama-3.2-1B Fine-tuned on Alpaca

This model is a LoRA-finetuned version of [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) on the Alpaca instruction-following dataset.

## Model Details

### Model Description

- **Developed by:** Parth Sinha
- **Model type:** Causal LM with LoRA adapters
- **Language(s):** English
- **License:** Meta Llama 3 Community License
- **Finetuned from model:** meta-llama/Llama-3.2-1B

### Model Sources

- **Repository:** [your GitHub repo](https://github.com/parthsinha1/fine-tuning-project)

## Uses

### Direct Use
- Instruction-following conversational AI, text generation, educational bots.

### Downstream Use
- Can be further fine-tuned for more specific tasks.

### Out-of-Scope Use
- Not suitable for safety-critical applications.

## Bias, Risks, and Limitations

- The model may reproduce biases present in the Alpaca dataset.
- Not evaluated for fairness or safety.

### Recommendations

- For research and educational use only.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/results/", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

prompt = "Explain what Python is."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

- [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca)

### Training Procedure

#### Preprocessing
- Instructions and responses formatted as:
  ```
  ### Instruction:
  <instruction>
  ### Response:
  <response>
  ```

#### Training Hyperparameters
- fp16 mixed precision
- batch size: 1
- gradient accumulation: 2
- learning rate: 1e-4
- epochs: 1

#### Speeds, Sizes, Times
- Trained on GTX 1080 Ti for ~20 minutes
- Model size: ~1B params + LoRA adapters

## Evaluation

### Testing Data, Factors & Metrics
- Manual prompt evaluation

### Results
- Good instruction following on test prompts

## Environmental Impact

- **Hardware Type:** GTX 1080 Ti
- **Hours used:** ~0.5
- **Cloud Provider:** Local
- **Compute Region:** N/A

## Technical Specifications

### Model Architecture and Objective
- Llama-3.2-1B with LoRA adapters for efficient fine-tuning.

### Compute Infrastructure
- OS: Windows 10
- Python: 3.12
- PyTorch: 2.3.0
- Transformers: 4.41.0
- PEFT: 0.16.0

## Citation

**BibTeX:**
```
@misc{parthsinha2025llama3alpaca,
  title={Llama-3.2-1B LoRA Fine-Tuned on Alpaca},
  author={Sinha, Parth},
  year={2025},
  url={https://github.com/parthsinha1/fine-tuning-project}
}
```

## Model Card Authors
- Parth Sinha

## Model Card Contact
- [your email or GitHub]
### Framework versions

- PEFT 0.16.0
