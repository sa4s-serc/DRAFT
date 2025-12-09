from dotenv import load_dotenv
import os
import vertexai
from vertexai.preview.tuning import sft

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
BASE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME=f"{BASE_MODEL}-TB-finetuned"

# Using TB datasets
TRAINING_URI = f"gs://{BUCKET_NAME}/datasets-ft/TBtrain.jsonl"
VALIDATION_URI = f"gs://{BUCKET_NAME}/datasets-ft/TBval.jsonl"

print(f"Starting Fine-Tuning Job for {BASE_MODEL} (Title-Body)...")

vertexai.init(project=GCP_PROJECT_ID, location=LOCATION)

tuning_job = sft.train(
    source_model=BASE_MODEL,
    train_dataset=TRAINING_URI,
    validation_dataset=VALIDATION_URI,
    tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
    epochs=5,
)

print(f"Tuning job submitted. Track status here: {tuning_job.resource_name}")
print("Waiting for tuning job to complete...")

TUNED_MODEL_NAME = getattr(tuning_job, "tuned_model_name", None)

if TUNED_MODEL_NAME:
    print("\n--- Fine-Tuning Complete ---")
    print(f"Tuned Model Resource Name: {TUNED_MODEL_NAME}")
else:
    print("\nFine-tuning is running asynchronously; the tuned model name is not yet available.")
    print(f"Track the job here: {tuning_job.resource_name}")