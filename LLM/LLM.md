# LLM selection

## Embedding Model

To implement the retrieval components for the RAFG and DRAFT approaches, we selected embedding models that align with our generative models:
- Open-Source: We used **Qwen3-Embedding-8B**, which ranks at the top of the **MTEB Retrieval leaderboard [https://huggingface.co/spaces/mteb/leaderboard], to pair with our open-source generative model. The rankings of METB as of 8 Sep 2025 are there in MTEB.csv.
- Proprietary: For the **Gemini-2.5-Flash** model, we used the corresponding **gemini-embedding-001** model from the same provider to ensure ecosystem compatibility.



## Generating model

For Generating LLMs we are Taking LLMs from **LM Arena text** rankings [https://lmarena.ai/leaderboard/text] as of 8 Sep 2025, as given in screencapture-lmarena-ai-leaderboard-text-2025-09-08-15_27_44.png

The selected models are:
- gpt-5-high
- gemini-2.5-flash
- qwen3-30b-a3b-instruct-2507
- gemma-3-4b-it
