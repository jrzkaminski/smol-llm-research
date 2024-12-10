# MetaGPT: The Multi-Agent Framework

> [!NOTE]
> https://arxiv.org/pdf/2308.00352

> [!TIP]
> https://github.com/geekan/MetaGPT


> [!IMPORTANT]
> TL;DR MetaGPT is a meta-programming framework leveraging LLM-based multi-agent collaboration with human-inspired SOPs for structured task execution. It outperforms previous systems like AutoGPT and ChatDev in generating robust, executable software solutions, achieving state-of-the-art results on benchmarks. The framework introduces structured communication, role specialization, and an iterative feedback mechanism, making it highly efficient and error-resistant.


## Introduction:

The paper introduces **MetaGPT**, a meta-programming framework designed for multi-agent collaboration based on large language models (LLMs). It is structured to address the shortcomings of existing LLM-based multi-agent systems that struggle with complex tasks due to logical inconsistencies and cascading errors. MetaGPT incorporates **Standardized Operating Procedures (SOPs)** to streamline workflows, ensuring roles and responsibilities mimic human team dynamics, which reduces errors and enhances collaboration.

## Key Features:
1. **Role Specialization:** Inspired by real-world workflows, MetaGPT assigns specific roles like Product Manager, Architect, Engineer, and QA Engineer to agents. Each role contributes structured outputs such as requirements documents, system designs, or test cases.
2. **Structured Communication:** It replaces free-form dialogues with structured, standardized outputs to prevent miscommunication and ensure clarity.
3. **Assembly Line Paradigm:** The framework decomposes complex tasks into subtasks that agents execute sequentially, following defined SOPs.
4. **Executable Feedback Mechanism:** A self-correction feature ensures that generated code is debugged and iteratively improved, addressing runtime issues.
5. **Global Message Pool:** A shared space allows agents to publish and subscribe to relevant task-related messages, improving efficiency without overwhelming agents with unrelated data.

## Performance:
- MetaGPT achieves state-of-the-art results on the HumanEval and MBPP benchmarks, with **Pass@1 rates of 85.9% and 87.7%**, respectively.
- It outperforms other frameworks like AutoGPT, LangChain, and ChatDev in generating executable and coherent software solutions.
- On the SoftwareDev dataset, MetaGPT demonstrated superior task completion, efficiency, and reduced human revision costs.

## Contributions:
1. Establishing MetaGPT as a flexible and efficient platform for developing LLM-based multi-agent systems.
2. Demonstrating the effectiveness of SOPs in enhancing task decomposition, collaboration, and output quality.
3. Introducing mechanisms like role-based management, structured communication, and executable feedback for robust code generation.

## Future Directions:
The paper discusses integrating self-referential learning mechanisms to allow agents to improve from past projects and potentially adopting dynamic economies among agents for better task collaboration.

---
