# ADR Data Processing Pipeline

This directory contains a comprehensive pipeline for collecting, processing, cleaning, and preparing a high-quality dataset of Architectural Decision Records (ADRs) from various software repositories, as given by previous study.

The process is broken down into five main stages:

1.  **Aggregate**: Gathers initial metadata about ADRs.
2.  **Scrape**: Downloads the raw ADR markdown files.
3.  **Dataset Creation**: Parses the raw files into a structured format.
4.  **Filter**: Cleans the dataset based on quality metrics.
5.  **De-duplicate**: Removes near-identical ADRs.

```mermaid
    A[Start: Raw Repository Info] --> B(1. Aggregate Metadata);
    B --> C{data.csv};
    C --> D(2. Scrape ADRs);
    D --> E{Raw .md Files};
    E --> F(3. Create Dataset);
    F --> G{adrs_output.jsonl};
    G --> H(4. Filter Data);
    H --> I{filtered_adrs_output.jsonl};
    I --> J(5. De-duplicate);
    J --> K[Final Dataset: adrs.jsonl];
    J --> L{Duplicates Log: removed_duplicates.jsonl};
```

---

## 1. Aggregate Metadata

This initial step processes JSON files located in the `repositories` folder. Each JSON file contains metadata for a single software repository.

* **Input**: JSON files containing repository information.
* **Process**:
    * Reads metadata for each repository, including its URL, ADR directories, and file paths.
    * Counts the occurrences of different ADR template types (`Nygard`, `Madr`, etc.).
* **Output**: A single CSV file, `ADR-data/data.csv`, which serves as a manifest for the scraping process. It includes columns for the repository URL and a delimited list of all ADR file paths within that repository.

---

## 2. Scrape ADR Files

Using the `ADR-data/data.csv` file, this stage downloads the actual markdown files for each ADR. This is handled by the `scrape.py` script.

* **Input**: `ADR-data/data.csv`.
* **Process**:
    * For each repository, the script attempts to download every ADR file.
    * It uses a robust fallback mechanism for determining the correct Git branch, trying `main`, then `master`, and finally querying the GitHub API for a list of all available branches if needed.
* **Output**: All downloaded ADR files are stored in the `all_ADRs` directory, organized into subfolders named after their respective repositories.

---

## 3. Create Dataset

This step transforms the raw, unstructured markdown files into a structured dataset.

* **Input**: Raw ADR markdown files from the `all_ADRs` directory.
* **Process**:
    * The script iterates through each ADR file to extract key sections.
    * **Title Extraction**: The title is identified using heuristics, such as checking for a `title` field in YAML frontmatter or parsing the first heading. If these methods fail, the filename is used as a fallback.
    * **Section Parsing**: It intelligently extracts the **Context** and **Decision** sections by searching for common headings (e.g., 'Context and Problem Statement', 'Decision Outcome').
    * **Context Fallback**: A key feature is its fallback logic. If an explicit "Context" section is not found, the script designates all content preceding the "Decision" section as the context.
* **Output**: A JSONL file, `ADR-data/adrs_output.jsonl`. Each line is a JSON object representing one ADR with fields like `Title`, `Body`, `Context`, `Decision`, and token counts. ADRs where a decision could not be extracted are skipped.

---

## 4. Filter

To ensure data quality, the structured dataset undergoes a rigorous filtering process.

* **Input**: `ADR-data/adrs_output.jsonl`.
* **Process**: Records are removed based on the following criteria:
    * **Token Count**: ADRs with bodies, contexts, or decisions that are too short or too long are dropped (e.g., body must be between 10 and 1000 tokens).
    * **Language**: Non-English ADRs are filtered out by checking if the text is composed of at least 90% printable ASCII characters.
    * **URL Density**: ADRs with a high ratio of URLs (more than 20% of tokens in the context or decision) are removed.
* **Output**: A cleaned JSONL file, `ADR-data/filtered_adrs_output.jsonl`.

---

## 5. De-duplicate

The final step removes redundant or near-identical ADRs from the dataset.

* **Input**: `ADR-data/filtered_adrs_output.jsonl`.
* **Process**:
    * The `Body` of each ADR is converted into a vector embedding using the `sentence-transformers` library with the `all-MiniLM-L6-v2` model.
    * The **cosine similarity** is calculated between every pair of ADR embeddings.
    * If the similarity score between two ADRs is **≥ 0.98**, one is marked as a duplicate and removed.
* **Output**:
    * `ADR-data/adrs.jsonl`: The final, clean, and de-duplicated dataset. ✅
    * `ADR-data/removed_duplicates.jsonl`: A log file containing the paths of the removed ADRs and which ADR they were a duplicate of.