train-baseline:
	@jobid=$$(sbatch --job-name=train-baseline \
		--output=logs/train-baseline_%j.out \
		run.slurm src.train_baseline \
		--task B \
		--output_dir outputs/baseline_test \
		--epochs 1 \
		--batch_size 64 \
		--learning_rate 2e-5 \
		--max_length 512 | awk '{print $$4}'); \
	echo "Submitted batch job $$jobid"; \
	logfile=logs/train-baseline_$${jobid}.out; \
	echo "Tailing $$logfile..."; \
	while [ ! -f $$logfile ]; do sleep 1; done; \
	tail -f $$logfile

get-jobs:
	@squeue -u "sugam.karki"
