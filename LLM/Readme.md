# LLM selection

## Embedding Model

To implement the retrieval components for the RAFG and DRAFT approaches, we selected embedding models that align with our generative models:
- Open-Source: We used **Qwen3-Embedding-8B**, which ranks at the top of the **MTEB Retrieval leaderboard [https://huggingface.co/spaces/mteb/leaderboard], to pair with our open-source generative model. The rankings of METB as of 8 Sep 2025 are there in MTEB.csv.
- Proprietary: We used an embedding model from the same proveider as that of the generative model to ensure ecosystem compatibility. We used the model 'recomended' by the corresponding LLM provider. For the **Gemini-2.5-Flash**, we used the corresponding **gemini-embedding-001** embedding model from Google. And for **gpt-5-high** we used **text-embedding-3-large** from OpenAI.


## Generating model

For Generating LLMs we are Taking LLMs from **LM Arena text** rankings [https://lmarena.ai/leaderboard/text] as of 8 Sep 2025, as given in screencapture-lmarena-ai-leaderboard-text-2025-09-08-15_27_44.png

The selected LLMs are:
- gpt-5-high
- gemini-2.5-flash
- qwen3-30b-a3b-instruct-2507
- gemma-3-4b-it

The selected Embedding models are:
- text-embedding-3-large
- gemini-embedding-001
- Qwen3-Embedding-8B
