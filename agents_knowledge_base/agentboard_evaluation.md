# AgentBoard: An Analytical Evaluation Board of Multi-Turn LLM Agents
https://arxiv.org/pdf/2401.13178
## Problem
The evaluation of Large Language Models (LLMs) as general-purpose agents poses significant challenges due to the following reasons:

1. **Benchmarking Complexity**: Current benchmarks fail to effectively evaluate agent performance across diverse scenarios in a unified framework. Many existing benchmarks focus on final success rates but lack insights into the step-by-step process.
2. **Lack of Depth in Analysis**: Simplistic metrics like success rates do not reveal finer distinctions or incremental progress in agent abilities. This is particularly inadequate for partially-observable, multi-turn environments, where agents must actively explore and adapt.
3. **Underdeveloped Evaluation Tools**: Current evaluation systems do not allow for detailed analyses such as tracking progress, breaking down performance by sub-skills, or understanding agentic limitations in realistic settings.

## Proposed Solution: AGENTBOARD
The authors introduce **AGENTBOARD**, a benchmark and open-source evaluation framework for the analytical assessment of multi-turn LLM agents. The framework aims to overcome the above challenges by offering:

1. **Diverse Tasks**:
   - Nine unique tasks covering four categories: embodied AI, games, web-based interactions, and tool operations.
   - A total of 1,013 scenarios designed to reflect realistic and complex environments.

2. **Innovative Metrics**:
   - A **fine-grained progress rate metric** to capture incremental task advancements, even if final goals are not achieved.
   - Uniform evaluation criteria for partially-observable, multi-turn tasks.

3. **Comprehensive Analysis Toolkit**:
   - Detailed performance breakdown by sub-skills such as planning, world modeling, grounding, memory, and spatial navigation.
   - Tools for assessing agent behavior, including interactive visualization of progress rates, trajectory analysis, and grounding accuracy.

4. **Interactive Visualization**:
   - An open-source web panel (via WandB) for real-time, interactive visualization of agent performance across various dimensions.

5. **Unified Framework**:
   - Consistent interfaces for diverse tasks and uniform metrics for progress and success evaluation.

## Key Insights from Experiments
1. **Progress Rate as a Superior Metric**:
   - The fine-grained progress rate outperforms success rates in discriminating between agent capabilities and revealing nuanced differences.

2. **Proprietary Models Dominate**:
   - GPT-4 significantly outperforms other models, demonstrating superior capabilities in planning and multi-turn interaction, though it still struggles with highly complex tasks.

3. **Code and Long-Context Proficiency Matters**:
   - Models with strong programming and reasoning capabilities, like DeepSeek-67b, perform better on agentic tasks compared to general-purpose open-weight LLMs.

4. **Challenges with Open-Weight Models**:
   - Open-weight LLMs exhibit deficiencies in key areas such as planning, world modeling, and long-range interaction.

5. **Exploration and Sub-Skill Gaps**:
   - Models fail to explore environments sufficiently, often missing critical information in partially-observable settings. Sub-skill analysis highlights significant gaps in retrospection and spatial navigation.

## Implications
AGENTBOARD represents a significant advancement in the evaluation of LLM agents by addressing gaps in current benchmarking systems. It facilitates deeper insights into agentic abilities, helping developers identify areas for improvement and design stronger, more capable LLM agents.

This comprehensive framework, with its diverse tasks and analytical tools, aims to accelerate the development and understanding of general-purpose LLM agents in real-world settings.