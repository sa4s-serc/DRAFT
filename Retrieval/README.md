# Retrieval Engine for Few-Shot Generation

This directory contains a script for the **retrieval component** of a Retrieval-Augmented Generation (RAG) pipeline. It finds and structures few-shot examples from a dataset to support two tasks:
**Context-to-Decision** and **Title-to-Body generation**. <br>
We are also storing the time to be used for efficiency later.


## ⚙️ Methodology

The script follows a the following workflow for the `train`, `val`, and `test` data splits:

1. **Embed**  
   Generates vector embeddings for **Context** or **Title** fields using the `Qwen/Qwen3-Embedding-8B` model.

2. **Index**  
   Builds an efficient **FAISS** vector index for rapid cosine similarity searches.

3. **Retrieve**  
   For each data point (the "anchor"), it queries the index to find the **top two most similar neighbors**.

4. **Structure**  
   Saves the **anchor** and its **two retrieved neighbors** into a new `.jsonl` file.


## 📁 Directory Structure

```
├── Data/ADR-data/
│   ├── train.jsonl      # Input training data
│   ├── val.jsonl        # Input validation data
│   └── test.jsonl       # Input test data
│
├── Retrieval/           # Output directory (created by the script)
│   ├── CDtrain.jsonl    # Output for Context->Decision (train)
│   ├── CDval.jsonl      # Output for Context->Decision (val)
│   ├── CDtest.jsonl     # Output for Context->Decision (test)
│   ├── TBtrain.jsonl    # Output for Title->Body (train)
│   ├── TBval.jsonl      # Output for Title->Body (val)
│   └── TBtest.jsonl     # Output for Title->Body (test)
│
└── Retrival.ipynb       # The main Jupyter Notebook
```

## Output Format (for each datapoint)
```
{
  "Anchor": {
    "PrimaryKey": 42,
    "Context": "The anchor's context text goes here...",
    "Decision": "The anchor's corresponding decision..."
  },
  "Retrieved": [
    {
      "PrimaryKey": 150,
      "Context": "Context from the first most similar item...",
      "Decision": "Decision from the first most similar item..."
    },
    {
      "PrimaryKey": 88,
      "Context": "Context from the second most similar item...",
      "Decision": "Decision from the second most similar item..."
    }
  ],
  "Time": 0.051
}
```