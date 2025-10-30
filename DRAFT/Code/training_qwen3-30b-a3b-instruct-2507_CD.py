import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv
from copy import deepcopy
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
cache_dir = "/research/ug/ug2k21dual/csd/adyansh.kakran/cache"
output_dir = "DRAFT/Output/"

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE")


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=HUGGINGFACE_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=bnb_config,
    device_map="auto",
    token=HUGGINGFACE_TOKEN
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,  # typical values: 8–128
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # common for attention layers
        "gate_proj", "up_proj", "down_proj"      # common for MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

"""
Data Preparation
"""
# Load dataset
data_files = {"train": "Retrieval/qwen3-embedding-8B/CDtrain.jsonl", "validation": "Retrieval/qwen3-embedding-8B/CDval.jsonl"}
dataset = load_dataset("json", data_files=data_files)

# Smaller dataset for testing
dataset['train'] = dataset['train'].select(range(100))  # Use a smaller subset for training
dataset['validation'] = dataset['validation'].select(range(20))  # Use a smaller subset for validation

def format_example(example):
    retrieved_contexts = [doc["Context"] for doc in example["Retrieved"]]
    retrieved_decisions = [doc["Decision"] for doc in example["Retrieved"]]
    context = example["Anchor"]["Context"]
    decision = example["Anchor"]["Decision"]

    messages = [
        {"role": "system", "content": "You are an expert software architect responsible for maintaining and thoroughly documenting all architectural decisions. You are writing an Architectural Decision Record for a software. Below are a few examples of Context and the corresponding Decision. Following the examples, provide only the ## Decision for the final ## Context provided by the user. Provide only the Decision in about 2-400 words. Do not add any explanations, introductions, or additional responses."},
        {"role": "user", "content": f"## Context: {retrieved_contexts[0]}"},
        {"role": "assistant", "content": f"## Decision: {retrieved_decisions[0]}"},
        {"role": "user", "content": f"## Context: {retrieved_contexts[1]}"},
        {"role": "assistant", "content": f"## Decision: {retrieved_decisions[1]}"},
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
        max_length=3072
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
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)


## Loss logging setup
loss_log_path = "DRAFT/Output/loss_log_qwen3-30b-a3b-instruct-2507_CD.jsonl"

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