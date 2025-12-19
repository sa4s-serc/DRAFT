from unsloth import FastLanguageModel  # Import Unsloth
import os
import torch
import json
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
from dotenv import load_dotenv
from copy import deepcopy
from peft import LoraConfig

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
cache_dir = "/research/ug/ug2k21dual/csd/adyansh.kakran/DRAFT/cache"
output_dir = "Finetune/Output/"

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    token=HUGGINGFACE_TOKEN,
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=int(32),  # typical values: 8–128
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # common for attention layers
        "gate_proj",
        "up_proj",
        "down_proj",  # common for MLP layers
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

"""
Data Preparation
"""
# Load dataset
data_files = {
    "train": "Retrieval/qwen3-embedding-8B/TBtrain.jsonl",
    "validation": "Retrieval/qwen3-embedding-8B/TBval.jsonl",
}
dataset = load_dataset("json", data_files=data_files)

# dataset["train"] = dataset["train"].select(
#     range(100)
# )
# dataset["validation"] = dataset["validation"].select(
#     range(20)
# )


def format_example(example):
    title = example["Anchor"]["Title"]
    body = example["Anchor"]["Body"]

    # Turn into chat-like input
    messages = [
        {"role": "system", "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Write the ADR corresponding to the ADR Title provided by the User. Provide only the ADR content in about 10-800 words. Do not add any additional responses—only the ADR content."},
        {"role": "user", "content": f"# Title: {title}"},
        {"role": "assistant", "content": body}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


# the formated text is stored in the "text" field
dataset = dataset.map(format_example)


# Tokenize
def tokenize_function(examples):
    # Removed padding="max_length". The data collator will handle dynamic padding,
    # which is much more efficient.
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,  # Let the data collator handle padding
    )


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)


# Data collator
# Unsloth's loader automatically sets tokenizer.pad_token = tokenizer.eos_token
# if it's not set, so this collator will work perfectly.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,  # Unsloth highly recommends bf16
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,  # parallel data loading
    dataloader_pin_memory=True,  # faster CPU-GPU transfer
    group_by_length=True,  # group similar length sequences

)

model = FastLanguageModel.for_training(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)


## Loss logging setup
loss_log_path = "Finetune/Output/loss_log_qwen3-30b-a3b-instruct-2507_TB.jsonl"


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        # if control.should_evaluate:
        control_copy = deepcopy(control)
        train_metrics = self._trainer.evaluate(
            eval_dataset=self._trainer.train_dataset, metric_key_prefix="train_epoch"
        )
        val_metrics = self._trainer.evaluate(
            eval_dataset=self._trainer.eval_dataset, metric_key_prefix="val_epoch"
        )

        # Print the metrics instead of logging to WandB
        print(f"Training set evaluation at epoch {state.epoch}:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value}")

        print(f"Validation set evaluation at epoch {state.epoch}:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value}")

        log_data = {
            "epoch": state.epoch,
            "train_loss": train_metrics["train_epoch_loss"],
            "val_loss": val_metrics["val_epoch_loss"],
        }
        with open(loss_log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        return control_copy


trainer.add_callback(CustomCallback(trainer))

def main():
    # Evaluate on training set
    train_eval_results = trainer.evaluate(
        eval_dataset=trainer.train_dataset, metric_key_prefix="train"
    )
    train_loss = train_eval_results["train_loss"]

    # Evaluate on validation set
    val_eval_results = trainer.evaluate()
    val_loss = val_eval_results["eval_loss"]

    # Print
    print(f"Epoch 0 - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
    log_data = {"epoch": 0, "train_loss": train_loss, "val_loss": val_loss}

    with open(loss_log_path, "w") as f:
        f.write(json.dumps(log_data) + "\n")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
