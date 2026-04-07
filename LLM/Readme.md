# LLM selection

### Generative Models

For Generating we selected LLMs from the [LM Arena text leaderboard](https://lmarena.ai/leaderboard/text) as of September 8 2025, as given in screencapture-lmarena-ai-leaderboard-text-2025-09-08-15_27_44.png

| Model | Provider | Size | Availability |
|---|---|---|---|
| GPT-5-high | OpenAI | — | Proprietary |
| Gemini-2.5-Flash | Google | — | Proprietary |
| Qwen3-30B-A3B-Instruct-2507 | Alibaba | 30B | Open source |
| Gemma-3-4B-it | Google | 4B | Open source |

> GPT-5 was evaluated with Prompting and RAFG only — not available for fine-tuning at the time of the study.

### Embedding Models

To implement the retrieval components for the RAFG and DRAFT approaches, we selected embedding models that align with our generative models:
- **Open-source:** We used **Qwen3-Embedding-8B**, which ranks at the top of the **MTEB Retrieval leaderboard [https://huggingface.co/spaces/mteb/leaderboard], to pair with our open-source generative model. The rankings of METB as of 8 Sep 2025 are there in MTEB.csv.
- **Proprietary:** We used an embedding model from the same proveider as that of the generative model to ensure ecosystem compatibility. We used the model 'recomended' by the corresponding LLM provider.

| Generative Model | Embedding Model |
|---|---|
| GPT-5-high | text-embedding-3-large (OpenAI) |
| Gemini-2.5-Flash | gemini-embedding-001 (Google) |
| Qwen3-30B / Gemma-3-4B | Qwen3-Embedding-8B (MTEB #1, open source) |