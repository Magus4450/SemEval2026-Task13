import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from src.utils import get_gpu_info
from src.train_contrastive import ContrastiveCodeBERT
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def collate_fn(batch, tokenizer, max_length):
    codes = [item["code"] for item in batch]
    ids = [item["ID"] for item in batch]
    encodings = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encodings["ids"] = ids
    return encodings


@torch.no_grad()
def predict(model_path, parquet_path, max_length=512, batch_size=16, device=None):


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & tokenizer
    print("Loading model")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = ContrastiveCodeBERT.from_pretrained(model_path).to(device)
    model.eval()

    # Stream parquet dataset (no memory blowup!)
    dataset = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)

    # Validate schema
    first_row = next(iter(dataset))
    if not {"ID", "code"}.issubset(first_row.keys()):
        raise ValueError("Parquet file must contain 'ID' and 'code' columns")

    # DataLoader for streaming batches
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
    )

    output_path = os.path.join(model_path, "preds.txt")
    # Open CSV and write incrementally
    with open(output_path, "w") as f:
        f.write("ID,prediction\n")

        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.softmax(outputs.logits, dim=-1)
            pred_labels = preds.argmax(dim=-1).cpu().numpy()

            for i, id_ in enumerate(batch["ids"]):
                f.write(f"{id_},{pred_labels[i]}\n")

    logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained CodeBERT model (streaming)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model folder")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to input parquet file with ID and code")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda")


    args = parser.parse_args()
    get_gpu_info()
    predict(
        args.model_path,
        args.parquet_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )