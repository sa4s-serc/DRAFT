# DRAFT

## Data Processing

This is a pipeline for collecting, processing, cleaning, and preparing a high-quality dataset of ADRss.

The process is broken down into five main stages:

1.  **Aggregate**: Gathers initial metadata about ADRs.
2.  **Scrape**: Downloads the raw ADR markdown files.
3.  **Dataset Creation**: Parses the raw files into a structured format.
4.  **Filter**: Cleans the dataset based on quality metrics.
5.  **De-duplicate**: Removes near-identical ADRs.

Look into the Data directory for details.