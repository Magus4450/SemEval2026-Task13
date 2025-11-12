train-baseline:
    python -m src.train_baseline --task 'B' --output_dir 'outputs/baseline_test' --epochs 1 --batch_size 64 --learning_rate 2e-5 --max_length 512