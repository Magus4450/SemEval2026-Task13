# SemEval2026-Task13


## Abstract

Distinguishing human-written code from machine-generated code has become increasingly important as large language models (LLMs) grow more capable. This paper presents our system for SemEval-2026 Task~13 Subtask~B, which focuses on multi-class authorship detection across ten major LLM families and human-written code. We investigate a hybrid training strategy that combines cross-entropy classification with contrastive and triplet-based metric learning. To address the extreme class imbalance in the dataset, we undersample the human-written class to 5\% of its original size. Our best model, which integrates contrastive learning with a weighted loss formulation, achieves a macro-F1 score of 0.3967, outperforming a strong UnixCoder baseline trained on the full dataset. These results highlight the effectiveness of representation-level objectives for capturing stylistic patterns across generator families and improving generalization under imbalanced conditions.

## Run

1. Install conda environment and install dependencies
```bash
    conda create -n semeval python=3.10
    conda activate semeval
    pip install -r requirements.txt
```

2. Train model
    - Train locally
    ```bash
    python -m src.train_contrastive --model_name microsoft/unixcoder-base \
		--task B \
		--output_dir outputs/contrastive_point9 \
		--epochs 4 \
		--batch_size 16 \
		--learning_rate 5e-5 \
		--max_length 512\
        --contrastive_weight 0.9
    ```
    - Train in university HPC
    ```bash
    make train-contrastive-B
    ```

3. Generate prediction
    - Predict locally
    ```bash
    python -m src.predict_contrastive -model_path outputs/contrastive_point9 \
		--parquet_path test.parquet \
		--batch_size 256 \
		--device cuda \
		--max_length 512
    ```
    - Predict in university HPC
    ```bash
    make predict-contrastive
    ```