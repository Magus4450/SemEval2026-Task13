train-ppt:
	@jobid=$$(sbatch --job-name=train-baseline \
		--output=logs/train-baseline_%j.out \
		run.slurm src.train_ppt train-adapters \
		--out_ai models/lora_ai \
		--out_human models/lora_human \
		--bs 32 \
		--grad_accum 4 \
		--lr 2e-4 \
		--load_in_8bit \
		--max_length 768\
		--epochs 2\
		--warmup_ratio 0.05 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/train-baseline_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile
	
train-baseline-B:
	@jobid=$$(sbatch --job-name=train-baseline \
		--output=logs/train-baseline_%j.out \
		run.slurm src.train_baseline \
		--model_name microsoft/unixcoder-base \
		--task B \
		--output_dir outputs/baseline_test \
		--epochs 1 \
		--batch_size 256 \
		--learning_rate 2e-5 \
		--max_length 1024 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/train-baseline_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile

train-contrastive-B:
	@jobid=$$(sbatch --job-name=train-contrastive \
		--output=logs/train-contrastive_%j.out \
		run.slurm src.train_contrastive \
		--model_name microsoft/unixcoder-base \
		--task B \
		--output_dir outputs/contrastive_point9 \
		--epochs 4 \
		--batch_size 16 \
		--learning_rate 5e-5 \
		--max_length 512\
        --contrastive_weight 0.9 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/train-contrastive_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile

train-baseline-A:
	@jobid=$$(sbatch --job-name=train-baseline \
		--output=logs/train-baseline_%j.out \
		run.slurm src.train_baseline \
		--task A \
		--output_dir outputs/baseline_A \
		--epochs 1 \
		--batch_size 256 \
		--learning_rate 2e-5 \
		--max_length 512 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/train-baseline_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile


	
predict-baseline:
	@jobid=$$(sbatch --job-name=predict-baseline \
		--output=logs/predict-baseline_%j.out \
		run.slurm src.predict \
		--model_path outputs/baseline_test \
		--parquet_path test.parquet \
		--batch_size 256 \
		--device cuda \
		--max_length 512 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/predict-baseline_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile

predict-contrastive:
	@jobid=$$(sbatch --job-name=predict-contrastive \
		--output=logs/predict-contrastive_%j.out \
		run.slurm src.predict_contrastive \
		--model_path outputs/contrastive_point7 \
		--parquet_path test.parquet \
		--batch_size 256 \
		--device cuda \
		--max_length 512 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/predict-contrastive_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile

get-jobs:
	@squeue -u "sugam.karki"
