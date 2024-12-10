# API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs

---

> [!NOTE]
> https://arxiv.org/pdf/2304.08244

> [!TIP]
> https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank

> [!IMPORTANT]
> TL;DR API-Bank is a benchmark for tool-augmented LLMs, distinct for its realistic and comprehensive evaluation of planning, retrieving, and calling APIs. It features 2,138 APIs across 1,000 domains, with a novel multi-agent system for cost-efficient data creation.

## Overview

**Introduction and Motivation**:  
The paper introduces **API-Bank**, a benchmark designed to evaluate and enhance tool-augmented Large Language Models (LLMs). Tool augmentation refers to enabling LLMs to interact with external tools like APIs to overcome inherent limitations such as outdated training data or inability to perform specific tasks. The benchmark focuses on addressing three research questions:
1. How effective are LLMs in using tools?
2. How can we improve their tool utilization capabilities?
3. What challenges exist in optimizing this capability?

**Core Components of API-Bank**:
- **Evaluation System**: API-Bank includes a functional evaluation system with 73 real APIs, 314 dialogues, and 753 API calls. It measures three abilities: *Call*, *Retrieve+Call*, and *Plan+Retrieve+Call*.
- **Training Dataset**: The dataset consists of 1,888 dialogues and 2,138 APIs spanning 1,000 domains. A novel multi-agent framework automates data generation, reducing annotation costs by 98%.

**Experimental Highlights**:
- LLMs like GPT-4 demonstrate superior planning and reasoning for API use but face challenges in API retrieval.
- A fine-tuned model, **Lynx**, based on Alpaca-7B, surpasses its predecessor by 26% in API call accuracy.

**Key Contributions**:
- API-Bank provides the most comprehensive and diverse evaluation and training benchmarks compared to existing datasets.
- Novel insights into challenges like API retrieval accuracy and input parameter adherence.
- Demonstrates the effectiveness of automated multi-agent frameworks in constructing high-quality training data.

**Limitations and Future Work**:
API-Bank currently focuses on English and a limited model scale. Future iterations aim to include multilingual capabilities and larger-scale models.

