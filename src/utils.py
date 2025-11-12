import os
import torch
import psutil
import platform
import logging

logger = logging.getLogger(__name__)

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} CUDA device(s).")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {name}")
            logger.info(f"  Total Memory: {total_mem:.2f} GB")
    else:
        logger.warning("No CUDA device detected.")

    try:
        cpu_name = platform.processor() or "Unknown CPU"
        cpu_count = os.cpu_count()
        cpu_freq = psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else None
        total_ram_gb = psutil.virtual_memory().total / 1024**3

        logger.info(f"CPU: {cpu_name}")
        logger.info(f"  Cores: {cpu_count}")
        if cpu_freq:
            logger.info(f"  Max Frequency: {cpu_freq:.2f} GHz")
        logger.info(f"  Total System Memory: {total_ram_gb:.2f} GB")
    except Exception as e:
        logger.warning(f"Could not fetch CPU info: {e}")
