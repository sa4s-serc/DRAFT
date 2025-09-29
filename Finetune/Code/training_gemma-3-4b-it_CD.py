import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from dotenv import load_dotenv
from copy import deepcopy

model_name = "google/gemma-3-4b-it"
cache_dir = "../cache"
output_dir = "Finetune/Output/"

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",
    dtype=torch.bfloat16,
    token=HUGGINGFACE_TOKEN
)


"""
Data Preparation
"""
# Load dataset
data_files = {"train": "Retrieval/CDtrain.jsonl", "validation": "Retrieval/CDval.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# # Smaller dataset for testing
# dataset['train'] = dataset['train'].select(range(100))  # Use a smaller subset for training
# dataset['validation'] = dataset['validation'].select(range(20))  # Use a smaller subset for validation

def format_example(example):
    context = example["Anchor"]["Context"]
    decision = example["Anchor"]["Decision"]

    # Turn into chat-like input
    messages = [
        {"role": "system", "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Give a ## Decision corresponding to the ## Context provided by the User. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."},
        {"role": "user", "content": f"## Context: {context}"},
        {"role": "assistant", "content": f"## Decision: {decision}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

# the formated text is stored in the "text" field
dataset = dataset.map(format_example)


# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=1024
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


"""
Training Setup
"""
# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="epoch",
    # learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    # warmup_steps=20,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)


## Loss logging setup
loss_log_path = "Finetune/Output/loss_log.jsonl"

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        control_copy = deepcopy(control)
        train_metrics = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train_epoch")
        val_metrics = self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="val_epoch")
        
        # Print the metrics instead of logging to WandB
        print(f"Training set evaluation at epoch {state.epoch}:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")
        
        print(f"Validation set evaluation at epoch {state.epoch}:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value}")
        
        log_data = {"epoch": state.epoch, "train_loss": train_metrics['train_epoch_loss'], "val_loss": val_metrics['val_epoch_loss']}
        with open(loss_log_path, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        return control_copy
    

trainer.add_callback(CustomCallback(trainer))



"""
Main Training loop
"""
def main():
    # Evaluate on training set
    train_eval_results = trainer.evaluate(eval_dataset=trainer.train_dataset, metric_key_prefix="train")
    train_loss = train_eval_results["train_loss"]

    # Evaluate on validation set
    val_eval_results = trainer.evaluate()
    val_loss = val_eval_results["eval_loss"]

    # Print
    print(f"Epoch 0 - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    log_data = {"epoch": 0, "train_loss": train_loss, "val_loss": val_loss}

    with open(loss_log_path, 'w') as f:
        f.write(json.dumps(log_data) + '\n')

    # Start training
    trainer.train()
    

if __name__ == "__main__":
    main()