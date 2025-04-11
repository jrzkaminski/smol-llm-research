# ToolLLM: Facilitating Large Language Models to Master 16,000+ Real-World APIs

> [!NOTE]
> https://arxiv.org/pdf/2307.16789

> [!TIP]
> https://github.com/OpenBMB/ToolBench

> [!IMPORTANT]
> TL;DR: ToolLLM introduces ToolBench, a dataset encompassing 16,000+ APIs, and trains ToolLLaMA, a fine-tuned model enabling advanced tool use and reasoning. Distinctly, it incorporates multi-tool tasks and a novel decision-tree-based reasoning approach (DFSDT), achieving competitive performance against proprietary models like GPT-4 and ChatGPT.

## Objective:
To address the limitations of open-source large language models (LLMs) in tool-use capabilities, the research introduces a comprehensive framework, ToolLLM, including data construction (ToolBench), training, and evaluation.

## Key Components:
1. **ToolBench Dataset**:
   - **API Collection**: Sourced from RapidAPI Hub, 16,464 REST APIs spanning 49 categories.
   - **Instruction Generation**: Diverse API instructions (single and multi-tool use cases) are crafted using ChatGPT.
   - **Solution Path Annotation**: Annotated paths with ChatGPT utilizing a depth-first search-based decision tree (DFSDT) for handling complex instructions.

2. **Model Training and Evaluation**:
   - Fine-tuned LLaMA on ToolBench, resulting in ToolLLaMA, equipped with a neural API retriever for recommending relevant APIs.
   - Evaluation through ToolEval, a ChatGPT-backed automatic evaluator measuring pass and win rates for tool-use tasks.

3. **Methodology**:
   - Enhanced reasoning with DFSDT to overcome traditional approaches' error propagation and limited exploration.
   - Integration of multi-tool and multi-step reasoning scenarios to mirror real-world complexities.

## Results:
1. **Performance**:
   - ToolLLaMA demonstrated parity with ChatGPT and competitive generalization abilities compared to GPT-4.
   - Strong performance in zero-shot scenarios, adapting effectively to unseen APIs with only their documentation.
   
2. **ToolEval**:
   - Reliable and scalable evaluation framework correlating highly with human evaluations.
   - Measures pass rate (task completion) and win rate (solution quality).

3. **Comparison with Existing Benchmarks**:
   - ToolBench surpasses others in scale, diversity, and practical applicability, incorporating real-world APIs and multi-tool scenarios.

4. **Out-of-Distribution Generalization**:
   - ToolLLaMA performed robustly on APIBench, validating its generalization to new domains.

---
