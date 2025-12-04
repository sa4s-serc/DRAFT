import openai
import time
from dotenv import load_dotenv
import os

load_dotenv()

# ----------------------------------------
# CONFIG
# ----------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # <-- replace with your API key

TRAIN_FILE_PATH = "train.jsonl"
VALID_FILE_PATH = "valid.jsonl"
MODEL_TO_FINE_TUNE = "gpt-4.1-nano"
# ----------------------------------------


def upload_file(path):
    print(f"Uploading file: {path}")
    with open(path, "rb") as f:
        response = openai.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Uploaded: {response.id}")
    return response.id


def create_finetune_job(train_file_id, valid_file_id):
    print("\nCreating fine-tune job…")

    job = openai.fine_tuning.jobs.create(
        model=MODEL_TO_FINE_TUNE,
        training_file=train_file_id,
        validation_file=valid_file_id
    )

    print(f"Fine-tune job created: {job.id}")
    return job.id


def stream_events(job_id):
    print("\nStreaming fine-tuning events (including epoch losses):\n")

    for event in openai.fine_tuning.jobs.list_events(job_id, stream=True):
        # Print raw event message
        print(event)

        # Extract losses (if present)
        if hasattr(event, "data") and isinstance(event.data, dict):
            metrics = event.data.get("metrics", {})
            if metrics:
                train_loss = metrics.get("train_loss")
                valid_loss = metrics.get("valid_loss")
                epoch = metrics.get("epoch")

                if epoch is not None:
                    print(f"\n📘 EPOCH {epoch}")
                    print(f"   Training Loss:   {train_loss}")
                    print(f"   Validation Loss: {valid_loss}\n")


def main():
    print("\n--- Fine-tune GPT-4.1 Nano ---\n")

    # Upload training/validation data
    train_file_id = upload_file(TRAIN_FILE_PATH)
    valid_file_id = upload_file(VALID_FILE_PATH)

    # Create fine-tuning job
    job_id = create_finetune_job(train_file_id, valid_file_id)

    # Stream logs until job completes
    stream_events(job_id)


if __name__ == "__main__":
    main()
