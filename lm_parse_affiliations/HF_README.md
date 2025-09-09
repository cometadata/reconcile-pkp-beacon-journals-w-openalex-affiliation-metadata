---
library_name: transformers
license: apache-2.0
base_model:
- Qwen/Qwen3-4B
---

# Affiliation Parsing LoRA

This model is a fine-tuned version of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) trained using Group Relative Policy Optimization (GRPO) for parsing and extracting author affiliations from academic paper content.

## Model Description

- **Base Model**: Qwen3-4B (4.0B parameters)
- **Training Method**: Group Relative Policy Optimization (GRPO) with LoRA
- **Task**: Author affiliation extraction and parsing from academic paper content
- **Training Data**: arXiv author affiliations dataset with PDF content and corresponding author/affiliation annotations

## Training Details

### Training Configuration
- **Training Algorithm**: GRPO with Direct Reward optimization (dr_grpo)
- **Learning Rate**: 1e-5 with cosine scheduler and 3% warmup ratio
- **Training Epochs**: 0.36 epochs completed
- **Batch Size**: 1 per device, 8 gradient accumulation steps
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.01
  - Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

### Training Metrics
- **Total Training Steps**: 13,516
- **Total Tokens Processed**: 62,074,442
- **Final Training Loss**: 0.075
- **Mean Reward**: 3.13
- **Answer Reward**: 2.21 ± 0.65
- **Format Reward**: 0.925 ± 0.16

### Hardware
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Training Time**: ~23.9 hours (86,125 seconds)
- **Precision**: bfloat16

## Reward Functions

The model was trained with two reward functions:

1. **Format Reward**: Evaluates whether the generated output follows the expected structured format for author and affiliation data (standardized 0-1 scale)
2. **Answer Reward**: Assesses the accuracy of extracted author names and affiliations compared to ground truth annotations

## Usage

The model processes academic paper content (up to ~6,000 tokens) and extracts structured author and affiliation information. It uses a system prompt that guides the model to parse author details from PDF content.

### Expected Input Format
The model expects PDF content from academic papers as input, truncated to approximately 6,000 tokens for processing efficiency.

### Training Data Processing
- **Max Prompt Length**: 7,000 tokens
- **Max Completion Length**: 2,000 tokens
- **Input Truncation**: PDF content truncated to 6,000 tokens during preprocessing

## Performance

The model achieved strong performance on formatting compliance:
- **Format compliance**: 92.5% of outputs follow the correct structured format
- **Content extraction**: Competitive performance on author and affiliation extraction tasks
- **Consistent output**: Low variance in format reward indicates reliable structured output generation

## Training Infrastructure

- **Cluster**: SLURM-managed HPC environment
- **Node**: Single node with 8 H100 GPUs
- **Memory**: 2.1TB total system memory
- **CUDA Version**: 12.8

## Limitations

- Trained specifically on academic paper content for affiliation extraction
- Input limited to ~6,000 tokens due to truncation during training
- Performance may vary on paper formats significantly different from arXiv content
- Reward metrics are not standardized between 0 and 1 (except format reward), making absolute performance assessment challenging

## Model Output

The model generates structured author and affiliation data extracted from academic paper content, following the format patterns learned during GRPO training with the specified reward functions.