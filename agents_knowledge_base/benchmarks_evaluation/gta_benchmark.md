# GTA: A Benchmark for General Tool Agents

---
> [!NOTE]
> https://arxiv.org/pdf/2407.08713

> [!TIP]
> https://github.com/open-compass/GTA

> [!IMPORTANT]
> TL;DR: GTA is a new benchmark for evaluating LLMs' tool-use capabilities in realistic, multimodal scenarios using human-written queries and real tools. Results show that even advanced models struggle, revealing critical weaknesses in reasoning, tool invocation, and argument formatting.

## **Problem**

1. **Gaps in Current Tool-Use Evaluations**:
   - Existing benchmarks (e.g., APIBench, ToolBench) rely on AI-generated queries, single-step tasks, and virtual tools, which fail to reflect the complexity of real-world scenarios.
   - AI-generated queries tend to be simplistic, with explicit tool-use instructions, limiting the evaluation of models' reasoning and planning skills.

2. **Limitations in Multimodal and Real-World Relevance**:
   - Current methods focus primarily on text-based interactions, lacking real-world multimodal challenges (e.g., images, tables).
   - They do not simulate the real-world scenarios where tool-use steps are implicit, and multiple tools need to be orchestrated for task completion.

3. **Performance Bottlenecks in LLMs**:
   - Mainstream LLMs, such as GPT-4, achieve below 50% accuracy in real-world tasks, with many models performing below 25%.
   - Argument prediction (i.e., correctly identifying parameters for tools) is a notable weak point.

4. **Lack of Comprehensive Benchmarks**:
   - Current benchmarks fail to assess end-to-end execution and reasoning in complex tasks that require multiple steps and tool invocations.

---

## **Proposed Solution: The GTA Benchmark**

1. **Core Features of GTA**:
   - **Human-Designed Queries**: Queries are human-written to reflect real-world objectives while keeping tool-use steps implicit. This challenges LLMs to reason, plan, and select tools effectively.
   - **Real Deployed Tools**: Includes 14 executable tools spanning perception (e.g., OCR, image description), operation (e.g., GoogleSearch, DrawBox), logic (e.g., Calculator, Solver), and creativity (e.g., TextToImage, ImageStylization).
   - **Multimodal Inputs**: Tasks incorporate image files (e.g., spatial scenes, tables, screenshots) as context, requiring models to process both textual and visual data.
   - **Complex Task Design**: 229 tasks were curated, requiring multi-step tool invocations, reasoning, and integration of outputs across tools.

2. **Dataset Construction**:
   - Human annotators expanded initial exemplars into diverse queries with varying contexts while maintaining implicit tool-use requirements.
   - Tasks were verified for feasibility, with step-by-step tool invocation sequences and correct final answers recorded for benchmarking.

3. **Evaluation Platform**:
   - Fine-grained evaluation metrics assess the entire workflow, including:
     - **Step-by-Step Accuracy**: Measures instruction-following, tool selection, argument correctness, and intermediate summarization.
     - **End-to-End Accuracy**: Evaluates the final result after all tool invocations and reasoning steps.

---

## **Key Findings from the Benchmark**

1. **Performance Insights**:
   - GPT-4 completed less than 50% of the tasks, while other models achieved less than 25%.
   - API-based models outperformed open-source ones, but argument prediction remained a bottleneck across all models.

2. **Behavioral Patterns in Models**:
   - Models exhibited distinct tendencies in tool invocation:
     - **Aggressive Models (e.g., Yi, Deepseek)**: Frequently invoked tools but with low accuracy due to poor instruction-following.
     - **Conservative Models (e.g., Qwen)**: Invoked tools cautiously, leading to fewer errors but missed opportunities.
     - **Neutral Models (e.g., GPT series)**: Balanced tool-use frequency with robust instruction-following, yielding the highest final accuracy.

3. **Challenges in Argument Prediction**:
   - Argument prediction errors significantly impacted overall accuracy, making it the weakest link in most models’ tool-use workflows.

---

## **Recommendations for Improvement**

1. **Enhancing Argument Prediction**:
   - Focus on improving models’ ability to generate correct arguments for tool invocations, including value and format accuracy.

2. **Fine-Tuning Techniques**:
   - ReAct-style and JSON-based instruction fine-tuning (as demonstrated with Agent-Flan) significantly improved format adherence and instruction-following capabilities.

3. **Broader Implications**:
   - Insights from the GTA benchmark can guide the development of general-purpose tool agents, enhancing their reasoning, planning, and execution in real-world tasks.

---

## **Significance of the GTA Benchmark**

The GTA benchmark addresses a critical gap in the evaluation of LLMs by introducing realistic, multimodal tasks that require reasoning and the effective use of diverse tools. By highlighting existing limitations and providing a platform for systematic improvement, it contributes to advancing the field of general-purpose AI agents capable of solving complex, real-world problems.