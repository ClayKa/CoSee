# CoSee: Project Overview

## 1. What is CoSee?

**CoSee** (“Collaborative Seeing”) is a multi-agent, multimodal reasoning framework built on top of a vision–language model (VLM), currently using **Qwen3-VL-4B-Instruct** as the backbone.  

The core idea is simple:

> *Multiple lightweight agents share a **visual whiteboard** (the Board), write and read structured notes grounded in images, and collaborate to answer complex multi-page document questions under a tight compute budget.*

Instead of relying on a single long chain‐of‐thought or expensive self-consistency over many samples, CoSee aims to improve performance on document/multi-image VQA by **structured collaboration** and **shared memory**, while keeping the system:

- **Compute-conscious** (small model, few agents, few rounds, bounded tokens),
- **Auditable** (all intermediate visual notes are explicit and inspectable),
- **Modular** (Board / Agent / Controller / Model wrapper are cleanly separated).

The initial target application is **multi-page document question answering** (DocVQA, slide decks, multi-image context), but the design should generalize to other multimodal tasks that require “looking around” and aggregating evidence from several views.

---

## 2. Motivation and problem setting

Modern VLMs can describe images very well, but they often behave like **single, monolithic reasoners**:

- One model instance receives all pages and the question,  
- It produces a long chain-of-thought,  
- Or we sample many times and hope that majority vote/self-consistency gives a good answer.

This has several limitations, especially for **document / multi-image VQA**:

1. **Poor division of labor**  
   - All capabilities (scanning many pages, reading small text, cross-checking answers) are entangled in a single prompt.  
   - There is no explicit notion of “this agent scans globally, that one reads local details”.

2. **No shared, structured memory**  
   - Intermediate observations are usually just linear text in a long context.  
   - It is hard to track *which page* or *which region* supported which intermediate conclusion.

3. **Compute inefficiency under long contexts**  
   - Multi-page documents can easily hit long context regimes.  
   - Self-consistency (sampling many times) linearly multiplies cost without giving interpretable collaboration.

4. **Weak interpretability of collaboration**  
   - Even multi-agent chat baselines often share a single text buffer with unstructured turns.  
   - There is no explicit visual workspace where agents write localized notes or point to concrete views.

**Problem we address**  
Given a multi-page document (or a small set of related images) and a question, how can we:

- Use a *small* VLM (4B scale),
- With *multiple specialized agents*,
- That coordinate via a **shared visual workspace**,
- To produce better answers **under a fixed token budget**,  
- While making every step of the reasoning traceable (which agent saw what, when, and how it contributed)?

CoSee is designed as a concrete answer to this question.

---

## 3. Key idea: a shared visual Board for Hogwild-style multimodal collaboration

The core insight behind CoSee is:

> Instead of treating the VLM as a single, long-running protagonist, we can treat it as **several lightweight roles** that act on a shared, structured memory — a **Board** that stores visual notes tied to specific views (pages / regions).

This leads to a simple but powerful pattern:

1. **Shared visual Board (central workspace)**  
   - The Board is a structured store of *cells*.  
   - Each cell has at least:
     - A **view** (e.g., page index and optional bounding box / region metadata),
     - A **content** string (short note, observation, hypothesis, or correction),
     - **Tags** (e.g., `["title", "table", "date"]`),
     - **Author** (which agent wrote it),
     - **Step** (which collaboration step produced it).
   - The Board can be rendered into a concise text summary (`Board.to_text`) with strong length controls (max cells per page, max total characters).

2. **Role-specialized agents**  
   - We instantiate multiple agents, all powered by the *same* VLM backbone (Qwen3-VL-4B-Instruct), but with different **role prompts**, e.g.:
     - **Global Scanner** – skims many pages to find candidate locations and rough structure.
     - **Detail Reader** – zooms in on candidate regions and extracts precise text/values.
     - **Cross-Checker** – verifies hypotheses using alternative pages/regions, looks for contradictions or missing evidence.
   - Each agent receives:
     - The relevant **images** for this round,
     - The **question**,
     - The current **Board summary** (`Board.to_text()`),
     - Its **role description**.
   - It outputs a **structured action** (JSON-like dict) describing what it did.

3. **Action schema instead of free-form chat**  
   - Instead of unstructured conversation, agents emit actions like:
     - `INSPECT` – inspect a specific page/region and add a new observation cell.
     - `LINK` – link an existing cell to another piece of evidence.
     - `HYPOTHESIZE` – propose an answer with supporting cell IDs.
     - `REVISE` – correct or refine a previous note or hypothesis.
   - This makes the Board both a **memory** and a **log of decisions**.

4. **Controller: multi-round collaboration under budget**  
   - A `CoSeeController` orchestrates T rounds:
     - Each round, it renders the Board summary,
     - Calls each agent’s `act(...)`,
     - Parses actions and updates the Board,
     - Checks for a confident `HYPOTHESIZE` with enough evidence,
     - Optionally stops early (before hitting max rounds).
   - The final answer is generated by a “coordinator” (could be a dedicated agent or a simple rule) based on:
     - The best hypothesis,
     - The supporting Board cells.

5. **Compute-conscious by design**  
   - We explicitly track token usage per question:
     - Inputs: images + question + Board summary + role prompt,
     - Outputs: short action descriptions.
   - We constrain:
     - Number of agents `M`,
     - Number of rounds `T`,
     - `max_new_tokens` per action,
     - Board summary length.
   - This yields a **bounded, predictable cost**, in contrast to naive self-consistency or long monolithic CoT.

---

## 4. How CoSee differs from common baselines

CoSee is meant to be compared against three broad families of baselines:

1. **Single-Agent Chain-of-Thought**  
   - One VLM instance receives all images and the question;  
   - It produces a long reasoning trace and final answer;  
   - No explicit division of roles or shared workspace;  
   - Hard to control which page/region it actually “attends to,” beyond latent attention maps.

2. **Self-Consistency (answer-only ensembling)**  
   - Same single agent is sampled N times with different randomness;  
   - Final answer = majority vote over N outputs;  
   - Often improves accuracy but multiplies compute cost linearly;  
   - No structured collaboration or shared intermediate memory.

3. **Multi-Agent Chat without a Board**  
   - Multiple agents (different role prompts) chat in plain text;  
   - They implicitly share a dialogue buffer, but there is no structured visual workspace;  
   - References to pages/regions are free-form, making analysis and control difficult.

**CoSee’s distinguishing features:**

- **Shared visual Board, not just text history**  
  - The Board stores *localized* visual notes tied to specific views;  
  - It can be analyzed post-hoc to understand coverage, redundancy, and corrections.

- **Action schema + explicit roles**  
  - Agents do not just “talk”; they perform actions that manipulate the Board;  
  - Each action has a clear semantics and can be used for analysis.

- **Designed for small models and small budgets**  
  - CoSee is intentionally tested with a 4B-scale VLM,  
  - The collaboration protocol is kept short and token-bounded,  
  - The goal is to get “multi-agent benefits” without requiring a huge backbone or massive sampling.

---

## 5. Current scope and assumptions

At this stage, CoSee makes several explicit assumptions:

1. **Task scope**  
   - Focus on **multi-page document and slide VQA**, where:
     - Each sample has multiple related page images (e.g., PDF pages, slides),
     - Questions require aggregating information across pages/regions.

2. **Backbone model**  
   - Use **Qwen3-VL-4B-Instruct** as the only backbone VLM.  
   - All agents share the same model weights; only the **role prompts** differ.

3. **No training or fine-tuning loop in this project**  
   - CoSee is an **inference-time collaboration framework**, not a new training recipe.  
   - We do not fine-tune the backbone; all behavior comes from prompting and control logic.

4. **Limited toolset and environment interactions**  
   - Agents do not call external tools beyond what the VLM inherently provides (e.g., no separate OCR engine, no external retrieval API).  
   - All reasoning is “in-model,” constrained to the images and the Board.

5. **Evaluation focus**  
   - Main metrics:
     - Answer quality (e.g., ANLS, EM, F1 depending on dataset),
     - Token budget per question (efficiency),
     - Collaboration patterns (coverage, redundancy, corrections) derived from Board traces.

---

## 6. Non-goals and explicit exclusions

To keep CoSee focused and tractable for a single-author project, we explicitly **do not** aim to:

- Propose a new dataset, benchmark, or metric for DocVQA.  
- Modify low-level attention kernels, KV cache mechanisms, or implement a new inference engine.  
- Perform large-scale training, RL, or supervised fine-tuning of the VLM.  
- Address safety, calibration, or general evaluation frameworks as primary contributions (those are out of scope for this project, except for basic reporting of error patterns).

These non-goals are important guardrails to keep the implementation and experimentation small-scale and feasible.

---

## 7. Project status and development phases (high level)

This repository is being developed with the following phased plan:

1. **Local environment and Qwen wrapper**
   - Set up Python environment(s) for local dev and remote GPU;
   - Download and locally verify `Qwen3-VL-4B-Instruct`;
   - Implement a `QwenVLClient` wrapper that:
     - Loads the model and processor from a local path,
     - Implements `build_messages` and `generate`.

2. **Core abstractions**
   - Implement `Board` with:
     - Cell schema,
     - `add_cell`,
     - `to_text` with robust length control.
   - Implement `Agent` base class and a `DummyAgent` for early testing.
   - Implement `CoSeeController` with:
     - Multi-round orchestration,
     - Action parsing,
     - Early stopping on `HYPOTHESIZE`.

3. **Real agents and dev-set experiments**
   - Connect real Qwen-based agents to the controller;
   - Create small dev subsets from DocVQA/SlideVQA for fast iteration;
   - Debug collaboration dynamics and prompt design.

4. **Full experiments and analysis**
   - Implement baselines (single-agent CoT, self-consistency, multi-agent chat);
   - Run full experiments on selected datasets;
   - Collect Board traces and analyze:
     - Evidence coverage,
     - Redundancy,
     - Corrections and failure modes.

5. **Paper and documentation**
   - Consolidate experimental results and figures;
   - Document CoSee’s design and findings in an ACL-style paper;
   - Provide code, configuration, and minimal instructions for reproduction.

---

## 8. How this overview should be used by code-assist tools

For code-assist tools (e.g., Codex, GitHub Copilot, ChatGPT in VS Code), this document serves as:

- A **high-level spec** of what CoSee is supposed to do, and what it is *not* supposed to do.
- A reference for:
  - Naming conventions (CoSee, Board, Agent, Controller, QwenVLClient),
  - The collaboration protocol (actions, rounds, Board updates),
  - The intended scope and constraints (multi-page DocVQA, small model, constrained budgets).
- A context source when generating or refactoring code in:
  - `cosee/models/qwen_vl_wrapper.py`,
  - `cosee/board.py`,
  - `cosee/agents.py`,
  - `cosee/controller.py`,
  - `cosee/eval/*` and `scripts/*`.

When interacting with this repository, tools should **prefer designs consistent with this overview** and avoid introducing out-of-scope complexity (such as new training pipelines, external tools, or features that contradict the compute-conscious design.)
