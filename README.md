# Synthetic Data Generation and Multi-Label Classification Using Large Language Models

## Research Overview

This repository contains the computational infrastructure and methodology for a novel approach to synthetic data generation and privacy-preserving text classification, leveraging state-of-the-art Large Language Models (LLMs).

## Synthetic Data Generation Methodology

### LLM Selection and Parameterization
We employed three prominent Large Language Models for synthetic data generation:
- Llama 2 (Meta AI, 2023)
- Llama 3
- Zephyr

#### Temperature Optimization Strategy
To ensure the generation of distinguishable synthetic posts, we implemented a rigorous temperature optimization process:
1. Randomly sampled 50 original posts
2. Generated synthetic variants across temperature ranges (0.5 - 1.0)
3. Calculated cosine similarity between original and synthetic posts
4. Selected the temperature producing the lowest average cosine similarity

This methodology guarantees that generated posts are:
- Sufficiently distinct from original content
- Non-linkable to source materials
- Preserving contextual essence

#### Synthetic Data Mapping
Our generation approach utilized a 1:3 mapping principle, where each original post serves as the source for three synthetic variants, each produced by a different LLM.

## Multi-Label Classification Model

### Model Configuration
- Base Model: Pre-trained RoBERTa
- Classification Type: Multi-label

#### Training Specifications
- Training Epochs: 50
- Batch Size: 8 (training and evaluation)
- Weight Decay: 0.01
- Dataset Split: 80% training, 20% testing

### Performance Evaluation
The multi-label classifier demonstrated robust performance across synthetic datasets, with performance metrics closely aligning with those of the original dataset.

#### Category-Level Insights
Performance variability was observed across different PII (Personally Identifiable Information) categories:
- Most categories showed high F1-scores
- Exceptions: Degree/Designation and Physical Appearance categories
  - Lower scores attributed to:
    * Fewer occurrences in synthetic data
    * Extensive potential value variations
    * Lack of consistent underlying patterns

## Technical Dependencies

### Required Libraries
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Scikit-learn

### Computational Requirements
- Python 3.8+
- Substantial GPU resources recommended

## Installation
Download the Anonymous repository and extract, then:
```bash
cd synthetic-data-generation
pip install -r requirements.txt
```

## Ethical Considerations

This research prioritizes:
- Data privacy
- Synthetic data utility preservation
- Minimal identifiability of source materials

## Limitations and Future Work

While our approach demonstrates promising results, potential avenues for future research include:
- Expanding LLM diversity
- Refining temperature optimization techniques
- Improving performance for challenging PII categories

## Citation

If you utilize this research or methodology, please reference our forthcoming publication.


**Disclaimer**: This research is conducted under strict ethical guidelines with explicit consent and anonymization protocols.


**Helper Commmands**:
torchrun --nproc_per_node 1 Llama2_post-rephraser.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6


