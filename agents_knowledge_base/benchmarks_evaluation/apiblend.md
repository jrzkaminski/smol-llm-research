# API-BLEND: A Comprehensive Corpora for Training and Benchmarking API LLMs

---

> [!NOTE]
> https://arxiv.org/pdf/2402.15491

> [!TIP]
> https://github.com/IBM/API-BLEND

> [!IMPORTANT]
> TL;DR: API-BLEND is a diverse and large-scale benchmark for training and evaluating LLMs on API-based tasks. Unlike previous synthetic datasets, it combines human-annotated and transformed real-world datasets to enhance diversity and generalization. It uniquely focuses on sequencing APIs for complex tasks and demonstrates superior out-of-domain generalization, making it distinct among benchmarks for tool-augmented LLMs.

## Overview

**API-BLEND** is a comprehensive dataset designed to train and benchmark large language models (LLMs) for tool and API usage. The dataset addresses the challenge of enabling LLMs to interact with APIs effectively, particularly for tasks like API detection, slot filling, and sequencing of API calls. API-BLEND was created using a hybrid approach that incorporates:
1. **Language Model-Assisted Generation**: Utilizing existing datasets like Schema-Guided Dialogue (SGD) and MultiWOZ to create API sequences.
2. **Grammar-Based Conversion**: Transforming datasets such as ATIS and SNIPS into API-compatible formats.
3. **Off-the-Shelf Datasets**: Integrating pre-existing datasets like ToolBench and API Bank without modifications.

### Features and Contributions:
1. **Diversity and Scale**: Includes 10 datasets from multiple domains (dialog systems, semantic parsing, digital assistants) with over 190k instances for training and testing.
2. **Task Coverage**: Supports three key API tasks:
   - API detection
   - Slot filling
   - Sequencing APIs to execute complex tasks
3. **Comparison to Synthetic Approaches**: While prior synthetic datasets suffer from biases and limited diversity, API-BLEND offers diverse, human-annotated examples, improving generalization, especially for out-of-domain (OOD) APIs.
4. **Evaluation Framework**: Models are evaluated using precision, recall, F1 scores, and Longest Common Subsequence (LCS) to assess API detection, slot filling, and sequence correctness.
5. **Benchmarks**: Demonstrates superior performance of API-BLEND-trained models compared to state-of-the-art tool-augmented LLMs in OOD settings.

### Experimental Results:
1. **In-Domain Performance**: API-BLEND-trained models achieved high precision and recall for API and parameter detection, outperforming other models on tasks derived from datasets like SeqATIS and SeqMultiWOZ.
2. **Out-of-Domain Generalization**: API-BLEND-trained models surpassed tool-augmented models like ToolLLaMA and Lynx in generalizing to new API tasks and datasets (e.g., ToolAlpaca, API Bank).
3. **Robustness**: Improved handling of complex scenarios involving multiple API calls and diverse domains.

### Observed Challenges:
1. **Parameter Normalization Issues**: Differences in parameter formats or unnormalized values led to mismatches despite semantically correct outputs.
2. **Similar Slot Names**: Overlapping parameter names across datasets caused minor errors.
3. **Specialized Slot Values**: Challenges arose in OOD tasks involving unique parameter types like SQL or mathematical functions.

### Conclusion:
API-BLEND represents a significant advancement in datasets for training tool-augmented LLMs, promoting better generalization and performance. Future directions include expanding to multilingual datasets and incorporating environmental interactions for embodied agents.
