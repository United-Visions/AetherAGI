# Mind Ingestion Module

## 1. Directory Overview

The `ingestion/` directory contains all the scripts and modules responsible for populating the AetherMind's `mind` with knowledge. Its primary role is to find, process, clean, and embed external data, preparing it for storage in the Pinecone vector database.

## 2. Current Capabilities

-   **Web Crawler (`web_crawler.py`):** A general-purpose web crawler for ingesting content from the internet. It is currently configured to use FireCrawl for efficient and clean data extraction.
-   **K-12 Ingestor (`k12_ingestor.py`):** A specialized script designed to crawl and process educational URLs, specifically targeting K-12 learning materials. It outputs clean Markdown, which is ideal for embedding.
-   **Processor (`processor.py`):** This module acts as the pipeline connector. It takes the output from the various crawlers (like FireCrawl), processes the text, generates embeddings, and handles the `.upsert()` operation into the Pinecone index. It is responsible for tagging data with the correct namespace (e.g., `core_k12`).
-   **Seed Axioms (`seed_axioms.py`):** A script to inject foundational truths and logical axioms into the `mind` as a baseline for the `brain`'s reasoning.

## 3. Interaction with Other Components

The `ingestion/` module is a preparatory system that works upstream of the core AetherMind loop.

-   **`mind/vector_store.py`:** The `processor.py` script directly interfaces with the `vector_store` to push new data into Pinecone. This is the final step of the ingestion process.
-   **External Services (FireCrawl, Web):** The crawlers in this directory are designed to interact with external web pages and services like FireCrawl to acquire raw data.
-   **Developers/Operators:** The scripts in this module are typically run manually or on a schedule by developers to expand AetherMind's knowledge base. They are not part of the real-time user interaction loop.

In essence, the `ingestion/` module is the set of tools used to "feed" AetherMind, ensuring its `mind` is constantly growing with high-quality, relevant, and well-structured information.
