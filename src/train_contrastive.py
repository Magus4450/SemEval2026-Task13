import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    RobertaModel,
    RobertaPreTrainedModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from src.utils import get_gpu_info
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import argparse
import logging
import warnings
import wandb


    
wandb.init(project="semeval2613_train_baseline")
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveCodeBERT(RobertaPreTrainedModel):
    """
    CodeBERT with contrastive learning for better feature separation
    between human-written and AI-generated code.
    """
    def __init__(self, config, contrastive_weight=0.3, temperature=0.07):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        self.projection_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(config.hidden_size, 256)
        )
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get CodeBERT embeddings
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # CLS token
        sequence_output = outputs[0][:, 0, :]
        
        # Classification head
        pooled_output = self.dropout(sequence_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Standard cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Contrastive loss
            if self.training and self.contrastive_weight > 0:
                # Project embeddings for contrastive learning
                projections = self.projection_head(sequence_output)
                projections = F.normalize(projections, p=2, dim=1)
                
                contrastive_loss = self.supervised_contrastive_loss(
                    projections, labels
                )
                
                # Combined loss
                loss = classification_loss + self.contrastive_weight * contrastive_loss
            else:
                loss = classification_loss
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised contrastive loss (SupCon).
        Pulls together samples of the same class and pushes apart different classes.
        """
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Mask out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        
        return loss


class CodeBERTTrainer:
    def __init__(self, task_subset='A', max_length=768, model_name="microsoft/codebert-base",
                 map_num_proc=None, contrastive_weight=0.3, temperature=0.07):      
        self.task_subset = task_subset
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None
        self.map_num_proc = map_num_proc
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
    def load_and_prepare_data(self):
        logger.info(f"Loading dataset subset {self.task_subset}...")
        try:
            dataset = load_dataset("DaniilOr/SemEval-2026-Task13", self.task_subset)
            train_data = dataset['train']
            logger.info(f"Loaded {len(train_data)} training samples")
            df = train_data.to_pandas()
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            logger.info(f"Sample data:\n{df.head()}")

            if 'code' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'code' and 'label' columns")

            df = df.dropna(subset=['code', 'label'])
            df['label'] = df['label'].astype(int)
            self.num_labels = df['label'].nunique()

            num_label0 = (df['label'] == 0).sum()
            print(f"Original label-0 count: {num_label0}")
            # keep_count = int(num_label0 // 2
            df_label0 = df[df['label'] == 0]
            df_label0_down = df_label0.sample(frac=0.05, random_state=42)
            df_other = df[df['label'] != 0]
            df = pd.concat([df_other, df_label0_down], ignore_index=True)

            # df = undersample(df)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            

            logger.info(f"Number of unique labels: {self.num_labels}")
            logger.info(f"Label range: {df['label'].min()} to {df['label'].max()}")
            logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

            # train_size = int(0.8 * len(df))
            # train_df = df[:train_size].reset_index(drop=True)
            # val_df = df[train_size:].reset_index(drop=True)
            train_df, val_df = train_test_split(df, test_size=0.05, stratify=df["label"])


            logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
            return train_df, val_df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def initialize_model_and_tokenizer(self, pretrained_path=None):
        logger.info(f"Initializing {self.model_name} model and tokenizer with contrastive learning...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        model_path = pretrained_path if pretrained_path else self.model_name
        
        # Load config
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = self.num_labels
        
        # Create contrastive model
        self.model = ContrastiveCodeBERT.from_pretrained(
            model_path,
            config=config,
            contrastive_weight=self.contrastive_weight,
            temperature=self.temperature
        )
        
        logger.info(f"Contrastive model initialized with {self.num_labels} labels")
        logger.info(f"Contrastive weight: {self.contrastive_weight}, Temperature: {self.temperature}")
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['code'],
            truncation=True,
            max_length=self.max_length
        )
    
    def prepare_datasets(self, train_df, val_df):
        logger.info("Preparing datasets for training...")

        train_dataset = Dataset.from_pandas(train_df[['code', 'label']])
        val_dataset = Dataset.from_pandas(val_df[['code', 'label']])

        num_proc = self.map_num_proc or max(1, (os.cpu_count() or 1) // 2)
        logger.info(f"Tokenizing with num_proc={num_proc}")

        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=num_proc
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['code'],
            num_proc=num_proc
        )

        train_dataset = train_dataset.rename_column('label', 'labels')
        val_dataset = val_dataset.rename_column('label', 'labels')

        train_dataset.set_format(type="torch")
        val_dataset.set_format(type="torch")

        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )

        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        try:
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                torch.cuda.synchronize(dev)

                allocated_mb = torch.cuda.memory_allocated(dev) / (1024 ** 2)
                reserved_mb  = torch.cuda.memory_reserved(dev)  / (1024 ** 2)
                peak_mb      = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

                metrics.update({
                    'gpu_mem_allocated_mb': round(allocated_mb, 1),
                    'gpu_mem_reserved_mb':  round(reserved_mb, 1),
                    'gpu_peak_mem_mb':      round(peak_mb, 1),
                })
        except Exception as e:
            logger.debug(f"GPU memory stats not available: {e}")

        return metrics

    
    def train(self, train_dataset, val_dataset, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5,
              dataloader_num_workers=None):   
        logger.info("Starting training...")

        if dataloader_num_workers is None:
            cpu = os.cpu_count() or 2
            dataloader_num_workers = max(2, min(8, cpu - 1))

        logger.info(f"Using dataloader_num_workers={dataloader_num_workers}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=1,
            dataloader_num_workers=dataloader_num_workers,     
            dataloader_pin_memory=True,                        
            dataloader_persistent_workers=True,
            report_to="wandb"            
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Training completed. Model saved to {output_dir}")
        return trainer
    
    def evaluate_model(self, trainer, val_dataset):
        logger.info("Evaluating model...")
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        logger.info("Classification Report:")
        print(classification_report(y_true, y_pred))
        return predictions
    
    def run_full_pipeline(self, output_dir="./results", num_epochs=3, batch_size=16, learning_rate=2e-5,
                          dataloader_num_workers=None):  
        try:
            train_df, val_df = self.load_and_prepare_data()
            self.initialize_model_and_tokenizer()
            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
            trainer = self.train(
                train_dataset, val_dataset,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dataloader_num_workers=dataloader_num_workers   
            )
            self.evaluate_model(trainer, val_dataset)
            logger.info("Pipeline completed successfully!")
            return trainer
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            raise


def main(): 
    parser = argparse.ArgumentParser(description='Train CodeBERT with Contrastive Learning on SemEval-2026-Task13')
    parser.add_argument('--model_name', default='microsoft/codebert-base', type=str, help='Name of the model')
    parser.add_argument('--task', choices=['A', 'B', 'C'], default='A', help='Task subset to use')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (increased for contrastive learning)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--map_num_proc', type=int, default=24, help='Workers for tokenization map()')  
    parser.add_argument('--loader_workers', type=int, default=None, help='DataLoader workers')
    parser.add_argument('--contrastive_weight', type=float, default=0.3, help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive learning')
    
    args = parser.parse_args()

    print(args)

    os.makedirs(args.output_dir, exist_ok=True)

    get_gpu_info()

    trainer = CodeBERTTrainer(
        task_subset=args.task,
        max_length=args.max_length,
        model_name=args.model_name,
        map_num_proc=args.map_num_proc,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature
    )
    
    trainer.run_full_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.loader_workers  
    )


if __name__ == "__main__":
    main()