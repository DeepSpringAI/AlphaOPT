# AlphaOPT Agentic Architecture

This document describes AlphaOPT's abstract agentic architecture as presented in
the reference paper, "AlphaOPT: Formulating Optimization Programs with a
Self-Improving LLM Experience Library," and as materialized in the codebase. It
avoids incidental implementation details while preserving the exact
agent-facing contracts: what each agent receives, what it returns, how agents
connect, and why the resulting system improves over time.

## 1. System Purpose

AlphaOPT translates natural-language optimization tasks into executable
Gurobi programs while continually improving a reusable experience library. It
is modeled after the workflow of optimization practitioners who learn recurring
modeling lessons from prior failures:

1. Retrieve relevant modeling and coding insights.
2. Build a mathematical formulation.
3. Translate the formulation into executable solver code.
4. Run the solver and compare the result with the known answer.
5. If the attempt fails, diagnose the formulation or program.
6. Extract reusable insights from the corrected attempt.
7. Verify that new insights are retrievable and useful.
8. Merge redundant insights.
9. Refine insight applicability conditions from aggregate evidence.

The key architectural idea is self-improving, solver-grounded experience
learning. AlphaOPT does not update model parameters. Instead, it builds a
structured library of reusable insights, each with explicit applicability
semantics, and uses solver feedback to decide which lessons are worth storing
and how their retrieval boundaries should evolve.

## 2. Global Orchestration

Let:

- \(t\): one optimization task.
- \(D_t\): the natural-language problem description.
- \(y_t\): the known optimal objective value.
- \(G_t\): the gold-standard program, when available.
- \(\ell\): the current experience library.
- \(\tau\): the hierarchical taxonomy dictionary.
- \(I_t^F\): formulation-stage insights retrieved for \(t\).
- \(I_t^P\): program-stage insights retrieved for \(t\).
- \(M_t\): the generated mathematical formulation.
- \(C_t\): the generated Gurobi program.
- \(z_t\): the objective value produced by executing \(C_t\).
- \(s_t\): the execution or optimality status.
- \(E_t\): diagnosed formulation or program issues.
- \(N_t\): newly extracted insights.
- \(V(N_t)\): the verification decision for new insights.
- \(\Pi(i)\): the task-interaction profile of insight \(i\).
- \(\ell^{*}\): the refined library used for downstream evaluation.

The high-level training orchestration is:

```text
(D_t, y_t, G_t, ell, tau)
  -> Library Retrieval
      outputs formulation insights I_t^F

(D_t, I_t^F)
  -> Formulation Generator
      outputs mathematical formulation M_t

(D_t, M_t, ell, tau)
  -> Library Retrieval
      outputs program insights I_t^P

(D_t, M_t, I_t^P)
  -> Program Generator + Solver Execution
      outputs C_t, z_t, s_t

(z_t, y_t, s_t)
  -> Optimality Checker
      outputs success/failure and feedback

if failure:
  (D_t, M_t, C_t, feedback, G_t or self-explored proxy)
    -> Diagnostic and Insight Extraction Agents
        output E_t, corrected artifacts, N_t

(N_t, ell, t)
  -> Retrieval-and-Success Verification
      outputs V(N_t)

verified N_t
  -> Merge and Library Update
      outputs updated ell, updated tau

after a diagnosis pass:
  ell with Pi(i)
    -> Library Evolution
        outputs refined ell*
```

The central semantic contract is the experience-library insight. Formulations
and programs are regenerated for each task, but durable learning happens
through structured insights whose conditions determine when they should be
retrieved again.

**Task record contract:**

Training and evaluation tasks are represented as task records with answer-level
supervision and optional expert artifacts:

```jsonc
{
  "task_id": "string or int",
  "description": "natural-language optimization problem",
  "ground_truth": float,
  "formulation": "optional reference formulation",
  "correct_program": "optional gold-standard solver program",
  "output_status": [],
  "success_count": int,
  "success_confidence": float,
  "fail_to_execute": int,
  "fail_to_verify": int,
  "retrieved_insights": [],
  "tag": "optional dataset tag",
  "cluster": "optional cluster label"
}
```

## 3. Agent Contracts

### 3.1 Experience Library

**Purpose:** Store solver-verified modeling and coding knowledge in a form that
can be retrieved, audited, merged, and refined.

**Input:**

- Newly verified insights \(N_t\).
- Existing library \(\ell\).
- Taxonomy dictionary \(\tau\).
- Per-task interaction evidence.

**Output:**

- Updated library \(\ell\).
- Updated taxonomy \(\tau\).
- Insight usage statistics and task-interaction distributions.

**Exact insight schema:**

Each stored insight is normalized to this contract:

```jsonc
{
  "insight_id": int,
  "taxonomy": {
    "Domain Modeling|General Formulation|Code Implementation": {
      "Level-1 label": ["Level-2 label"]
    }
  },
  "condition": "string",
  "explanation": "string",
  "example": "string",
  "task_id": "source task id",
  "iteration": int,
  "merge_version": int,
  "refine_version": int,
  "occurrence": [int],
  "correctness": [int],
  "distribution": {
    "positive": [],
    "negative": [],
    "unretrieved": [],
    "irrelevant": [],
    "invalid": []
  }
}
```

The conceptual insight representation in the paper is the four-tuple:

\[
i = (\text{taxonomy}, \text{condition}, \text{explanation}, \text{example}).
\]

The implementation adds identifiers, provenance, merge/refinement versions,
usage counters, and the distribution profile \(\Pi(i)\).

**Taxonomy contract:**

The library is organized under three top-level tracks:

```jsonc
{
  "Domain Modeling": {
    "Level-1 problem-domain label": {
      "Level-2 technique label": {
        "definition": "string",
        "condition": "string"
      }
    }
  },
  "General Formulation": {
    "Level-1 formulation-component label": {
      "Level-2 pitfall label": {
        "definition": "string",
        "condition": "string"
      }
    }
  },
  "Code Implementation": {
    "Level-1 coding-area label": {
      "Level-2 implementation issue label": {
        "definition": "string",
        "condition": "string"
      }
    }
  }
}
```

**How it works:**

The library is not a flat memory buffer. Its taxonomy first narrows the search
space, and insight conditions then decide applicability. New insights can add
new Level-1 or Level-2 labels when no existing label fits. Redundant insights
are merged only when their underlying OR principle is materially equivalent.

### 3.2 Library Retrieval Agent

**Purpose:** Retrieve only the insights that are relevant to a task or to a
specific diagnosed issue.

**Input:**

- Problem description \(D_t\).
- Current library \(\ell\).
- Taxonomy dictionary \(\tau\).
- Stage indicator: `Formulation`, `Program`, or `Diagnosis`.
- For program-stage retrieval, the generated mathematical formulation \(M_t\).
- For diagnosis retrieval, a diagnosed issue \(e \in E_t\).

**Output:**

- Candidate insights selected by taxonomy.
- Final applicable insights after condition checking.

Formally:

\[
\mathcal{R}_{F}(D_t, \ell, \tau) \rightarrow I_t^F,
\]

\[
\mathcal{R}_{P}(D_t, M_t, \ell, \tau) \rightarrow I_t^P.
\]

**Exact quick-match schema:**

For formulation-stage retrieval, the quick matcher returns taxonomy labels:

```jsonc
{
  "Domain Modeling": {
    "Resource Allocation": ["Capacity/Resource Balance Equations"]
  },
  "General Formulation": {
    "Variable Definition": ["Continuous vs. Discrete Confusion"]
  }
}
```

For program-stage retrieval, the same shape is restricted to code labels:

```jsonc
{
  "Code Implementation": {
    "Solver & API Syntax": ["Strict Inequalities"]
  }
}
```

If no labels apply, the contract is:

```json
{}
```

**Exact applicability-check schema:**

After taxonomy matching, candidate insights are checked one by one. The final
output is:

```jsonc
[
  {
    "insight_id": 1,
    "reason": "1-2 sentence applicability justification."
  }
]
```

If no insight is applicable, the contract is:

```json
[]
```

**How it works:**

Retrieval is deliberately two-stage. The taxonomy pass cheaply identifies
plausible regions of the library. The condition pass evaluates each candidate
against the actual task context and removes insights whose inapplicability
clauses would make them misleading. This is AlphaOPT's main defense against
negative transfer.

### 3.3 Formulation Generator

**Purpose:** Produce a formal mathematical optimization model from a natural
language task, optionally guided by formulation-stage insights.

**Input:**

- Problem description \(D_t\).
- Formulation insights \(I_t^F\), drawn from `Domain Modeling` and
  `General Formulation`.

**Output:**

- Mathematical formulation \(M_t\).

Generation mapping:

\[
\mathcal{G}_{F}: (D_t, I_t^F) \rightarrow M_t.
\]

**Exact artifact contract:**

The generator must return only one plain fenced block with four sections:

````text
```
## Parameters:
[fixed data, constants, symbols, units, sets, and indices]

## Variables:
[decision variables, meanings, domains, and types]

## Objective:
[objective expression]

## Constraints:
[labeled equations and inequalities]
```
````

When a retrieved insight is actually used, the formulation may annotate the
relevant component with a brief comment identifying the insight ID and how it
helped. If no insight applies, no invented insight annotation is allowed.

**How it works:**

The formulation generator is the first place where retrieved experience
changes the solving behavior. It uses insights as soft, task-specific guidance
for avoiding known modeling mistakes, but the problem description remains the
source of truth.

### 3.4 Program Generator

**Purpose:** Translate a mathematical formulation into a complete executable
Gurobi program and run it.

**Input:**

- Problem description \(D_t\).
- Mathematical formulation \(M_t\).
- Program-stage insights \(I_t^P\), drawn from `Code Implementation`.

**Output:**

- Executable Python/Gurobi program \(C_t\).
- Solver output \(z_t\) or execution feedback.
- Runnable and timeout flags.

Generation mapping:

\[
\mathcal{G}_{P}: (D_t, M_t, I_t^P) \rightarrow C_t.
\]

Execution mapping:

\[
\text{Exec}(C_t) \rightarrow (z_t, \text{runnable}, \text{timeout}).
\]

**Exact artifact contract:**

The program generator must output exactly one Python code block:

```python
import gurobipy as gp
from gurobipy import GRB
model = gp.Model("OptimizationProblem")
# generated model code
model.optimize()
```

The implementation appends a standard post-solve footer that prints:

```text
Optimal value: <model.ObjVal>
Model is infeasible.
Model is unbounded.
Other status: <status>
```

The solver wrapper extracts the numeric objective from `Optimal value:` when it
is present.

**How it works:**

The program generator is constrained to follow \(M_t\) rather than silently
inventing a different formulation. Code insights target implementation-level
pitfalls such as solver API syntax, strict inequalities, nonlinear expression
handling, variable bounds, and data/index alignment.

### 3.5 Optimality Checker

**Purpose:** Convert raw solver execution into a task-level success signal.

**Input:**

- Solver output \(z_t\).
- Ground-truth optimal objective \(y_t\).
- Runnable flag.
- Timeout flag.

**Output:**

- Boolean optimality decision.
- Status string.
- Feedback used by downstream diagnostic agents.

Validation rule:

\[
\text{Success}(t \mid \ell) =
\mathbf{1}\{|z_t - y_t| \leq \epsilon\},
\]

when \(z_t\) is numeric and the program ran successfully.

**Exact status contract:**

```jsonc
{
  "is_optimal": bool,
  "status": "optimal|not_optimal|failure_solve|solver_time_out|run_error",
  "feedback": "string or null"
}
```

**How it works:**

AlphaOPT uses answer-level supervision. A task is considered solved when the
generated program produces an objective value within the configured tolerance
of \(y_t\). If the program is non-runnable or returns no numeric objective, the
status is routed to diagnostic logic rather than treated as a useful solution.

### 3.6 Program Diagnostic Agent

**Purpose:** Repair non-runnable programs and extract code-level lessons from
the repair.

**Input:**

- Failed program \(C_t\).
- Execution feedback.
- Mathematical formulation \(M_t\).
- Retrieved program insights \(I_t^P\).

**Output:**

- Corrected program.
- Updated execution status.
- Positive/negative labels for retrieved program insights.
- Optionally, unretrieved program insights discovered during diagnosis.

**Exact corrected-program contract:**

For each issue, the corrected program uses fix blocks:

```python
#===
# <brief issue explanation>
# wrong attempt: <incorrect code>
<corrected code>
#===
```

The full output is still a single Python code block ending with
`model.optimize()`, with no code modified after that line.

**Exact program-insight diagnosis schema:**

```jsonc
[
  {
    "insight_id": 12,
    "state": "positive|negative"
  }
]
```

**How it works:**

If the generated program fails at runtime, AlphaOPT first asks whether any
retrieved code insights were misleading. Negative insights are removed and the
program is regenerated. If the error persists, the diagnostic agent repairs the
code directly, producing fix blocks that the insight extractor can convert into
new `Code Implementation` insights.

### 3.7 Formulation Diagnostic Agent

**Purpose:** Diagnose why a runnable program or formulation fails to achieve the
known optimum, and determine whether retrieved insights helped or harmed.

**Input:**

- Problem description \(D_t\).
- Failed mathematical formulation \(M_t\).
- Execution feedback.
- Gold-standard program \(G_t\), or a self-explored proxy when \(G_t\) is
  absent.
- Retrieved formulation insights \(I_t^F\).

**Output:**

- A list of root-cause formulation issues.
- Positive/negative/invalid/irrelevant evidence for retrieved insights.
- Candidate unretrieved insights that could resolve diagnosed issues.

**Exact issue schema:**

```jsonc
[
  {
    "id": 1,
    "issue": "concise root-cause issue",
    "evidence": "#wrong ... #correct ..."
  }
]
```

**Exact retrieved-insight diagnosis schema:**

The prompt distinguishes four conceptual labels:

- `positive`: correct, adopted, and helpful.
- `invalid`: correct but not adopted; would have helped.
- `negative`: wrong or inapplicable, and risky or harmful.
- `irrelevant`: not relevant to the task.

The operational output consumed by the pipeline is:

```jsonc
[
  {
    "insight_id": 1,
    "state": "positive|negative",
    "evidence": "string"
  }
]
```

**Exact issue-validation schema:**

When testing whether a regenerated formulation resolves diagnosed issues, the
contract is:

```jsonc
[
  {
    "id": 1,
    "status": "solved|unsolved",
    "evidence": "string"
  }
]
```

**How it works:**

The formulation diagnostic agent compares a failed model with the correct
model embodied by a gold-standard or self-explored program. It reports
independent root-cause defects, not surface symptoms. It also separates two
different failure modes: an insight may be harmful because it was retrieved
when it should not have been, or a useful insight may be absent because its
condition was too narrow.

### 3.8 Self-Exploration Agent

**Purpose:** Create a solver-verified proxy gold-standard program when a task
has only answer-level supervision.

**Input:**

- Problem description \(D_t\).
- Failed programs and feedback.
- Known optimal objective value \(y_t\).

**Output:**

- Corrected program \(G_t'\) that is runnable and reaches \(y_t\), if found.
- Failure if the exploration budget is exhausted.

Self-exploration mapping:

\[
\mathcal{X}: (D_t, y_t, \{C_t^{(k)}, \text{feedback}^{(k)}\})
\rightarrow G_t'.
\]

**Exact artifact contract:**

The agent must output only a single Python code block with a top-level
`model.optimize()` call and no natural-language text outside the block.

**How it works:**

Self-exploration lets AlphaOPT learn from datasets that provide final answers
but no expert program. The explored program is not accepted merely because it
matches the answer; the prompt explicitly instructs the agent not to fabricate
data or violate the problem statement in order to force the objective value.

### 3.9 Insight Extractor

**Purpose:** Convert corrected formulations or corrected code into structured,
reusable insights.

**Input:**

- Problem description \(D_t\).
- Failed formulation \(M_t\) and correct program \(G_t\), for formulation
  insights.
- Corrected program with fix blocks, for code insights.
- Current taxonomy dictionary \(\tau\).

**Output:**

- New insight candidates \(N_t\).
- Optional taxonomy additions.

Extraction mapping:

\[
\mathcal{I}: (D_t, M_t, G_t, \tau) \rightarrow N_t^F,
\]

\[
\mathcal{I}: (M_t, C_t^{\text{fixed}}, \tau) \rightarrow N_t^P.
\]

**Exact formulation-insight schema:**

```jsonc
[
  {
    "taxonomy": {
      "Domain Modeling": {
        "Network Flow": {
          "Flow Conservation": null
        }
      }
    },
    "condition": "This insight applies when ... For example, when ...",
    "explanation": "When the problem involves ... The best practice is ...",
    "example": "# Wrong ... # Correct ..."
  }
]
```

For `Domain Modeling` and `General Formulation`, the extractor chooses exactly
one top-level stage, one Level-1 label, and one Level-2 label per insight. If a
new label is needed, it supplies:

```jsonc
{
  "definition": "one sentence",
  "condition": "one sentence"
}
```

**Exact program-insight schema:**

```jsonc
[
  {
    "taxonomy": {
      "Code Implementation": {
        "Solver & API Syntax": {
          "Strict Inequalities": null
        }
      }
    },
    "condition": "This insight applies when the mathematical model contains...",
    "explanation": "When the problem involves ... The best practice is ...",
    "example": "# Wrong ... # Correct ..."
  }
]
```

**How it works:**

The extractor distills a concrete failure into a general but bounded lesson. A
valid insight must say what context triggers it, what principle it teaches, and
what a wrong-versus-correct implementation looks like. This structure is what
makes later retrieval auditable rather than opaque.

### 3.10 Insight Verification Agent

**Purpose:** Ensure that newly generated insights are both useful and
retrievable before they enter the persistent library.

**Input:**

- Candidate new insights \(N_t\).
- Previous insights retrieved for the task.
- Temporary library containing the candidate insights.
- Source task \(t\).

**Output:**

- Full verification, partial verification, or rejection.
- Retrieval diagnostics for missed taxonomy or condition matches.

**Exact retrieval-verification schema:**

```jsonc
{
  "all_retrieved": bool,
  "retrieved_insight_ids": [int],
  "missed_insight_ids": [int],
  "taxonomy_failed": bool,
  "condition_failed": bool,
  "taxonomy_failed_insight_ids": [int],
  "condition_failed_insight_ids": [int]
}
```

**Verification rule:**

First, AlphaOPT replays the source task using the previous insights plus the
new insights:

\[
\text{Replay}(D_t, I_t \cup N_t) \rightarrow (z_t', s_t').
\]

The candidate set can be accepted only if:

\[
|z_t' - y_t| \leq \epsilon.
\]

Second, AlphaOPT checks whether the new insights can be retrieved again from
the temporary library:

\[
\text{Retrieve}(D_t, \ell \cup N_t) \supseteq N_t.
\]

Full verification requires both task success and retrieval of all new insights.
If the task succeeds but only some insights are retrieved, AlphaOPT may keep the
verified subset and try to rewrite missed insights.

**How it works:**

This stage prevents the library from accumulating inert or unfindable advice.
An insight is not useful to AlphaOPT merely because it is true; it must also be
recoverable by the retrieval mechanism under the source task context.

### 3.11 Insight Merge Agent

**Purpose:** Bound library growth by merging redundant insights without losing
the underlying verified principle.

**Input:**

- New verified insights.
- Existing library insights.
- Task-level or batch-level candidate groups.

**Output:**

- Empty result when no merge is appropriate.
- Merged canonical insights when redundancy is found.

**Exact batch-merge schema:**

```jsonc
[
  {
    "merged_ids": [1, 4],
    "reason": "why merging reduces redundancy",
    "taxonomy": {
      "General Formulation": {
        "Variable Definition": {
          "Explicit Bounds": null
        }
      }
    },
    "condition": "This insight applies when ...",
    "explanation": "When the problem involves ...",
    "example": "# Wrong ... # Correct ..."
  }
]
```

If no merge is appropriate:

```json
[]
```

**Exact online-merge schema:**

```jsonc
{
  "merged_ids": [12],
  "reason": "why merging reduces redundancy",
  "taxonomy": { },
  "condition": "string",
  "explanation": "string",
  "example": "string"
}
```

If no merge is appropriate:

```json
{}
```

**How it works:**

Merging is itself verified. A merged insight must still replay successfully on
the source tasks covered by the original insights. This lets AlphaOPT reduce
library complexity without weakening task success.

### 3.12 Library Evolution Agent

**Purpose:** Refine insight applicability conditions using aggregate evidence
across tasks.

**Input:**

- Library \(\ell\).
- For each insight \(i\), its task distribution:

\[
\Pi(i) = \{S_i^+, S_i^-, S_i^u\},
\]

where:

- \(S_i^+\): tasks where \(i\) was useful.
- \(S_i^-\): tasks where \(i\) was misleading.
- \(S_i^u\): tasks where \(i\) was not retrieved but would have been useful.

**Output:**

- Candidate refined conditions.
- Best accepted refined condition.
- Updated library \(\ell^{*}\).

**Exact negative-condition schema:**

```jsonc
{
  "condition": "This insight does NOT apply when ... For example, when ...",
  "reason": "1-2 sentence justification"
}
```

If the insight is actually applicable:

```json
{}
```

**Exact unretrieved-condition schema:**

```jsonc
{
  "condition": "This insight applies when ... For example, when ...",
  "reason": "1-2 sentence justification"
}
```

If the insight is not applicable:

```json
{}
```

**Exact refinement-variant schema:**

```jsonc
[
  {
    "path_id": 1,
    "strategy": "1-2 sentence description",
    "new_condition": "This insight applies when ...\n\nThis insight does NOT apply when ..."
  }
]
```

**Refinement score:**

For insight \(i\), AlphaOPT evaluates a candidate condition on:

\[
R_i = S_i^+ \cup S_i^- \cup S_i^u.
\]

The paper's refinement score is:

\[
p_i =
\frac{
|\text{kept positives}|
+
|\text{corrected negatives}|
+
|\text{recovered unretrieved}|
}
{|R_i|}.
\]

The best candidate is the one that improves this score most, by preserving
positive retrievals, suppressing negative retrievals, and recovering missed
retrievals.

**How it works:**

Library Evolution turns locally verified insights into more transferable
knowledge. Negative tasks add exclusion boundaries. Unretrieved tasks add
missing positive triggers. The refined condition is accepted only after replay
against the relevant task set, so refinement is driven by aggregate behavior
rather than isolated prompt editing.

## 4. Training Phases

AlphaOPT training is organized into a continual two-phase cycle, with an
initial online-learning pass followed by diagnosis and refinement iterations.

### 4.1 Library Online Learning

The online-learning phase processes training tasks in minibatches. For each
task, AlphaOPT:

1. Retrieves formulation insights.
2. Generates a formulation.
3. Retrieves program insights.
4. Generates and executes a Gurobi program.
5. Checks the objective against the answer label.
6. For failures, diagnoses and extracts new insights.
7. Self-verifies new insights through task replay and retrieval replay.
8. Merges verified insights at task, batch, and online-library levels.
9. Adds accepted insights to the persistent library.

This phase expands \(\ell\) while enforcing solver-grounded admission criteria.

### 4.2 Library Diagnosis

The diagnosis phase reruns training tasks with the current library and records
how insights interact with each task. For a retrieved insight \(i\), a task can
be added to:

- \(S_i^+\) when the insight is helpful.
- \(S_i^-\) when the insight is misleading.
- \(S_i^u\) when the insight was missed but later shown useful.

If removing a negative insight or adding an unretrieved insight solves the
task, AlphaOPT attributes the failure to applicability misalignment rather than
missing knowledge.

### 4.3 Library Refinement

The refinement phase uses \(\Pi(i)\) to improve the condition of each candidate
insight. It generates multiple candidate condition rewrites, builds temporary
library variants, replays retrieval on positive, negative, and unretrieved task
sets, and keeps the best variant when it improves the refinement score.

The accepted result becomes the library for the next iteration.

## 5. Evaluation-Time Flow

At evaluation time, AlphaOPT does not extract, merge, or refine new insights.
It uses an archived library:

```text
(D_t, ell*)
  -> retrieve formulation insights
  -> generate formulation
  -> retrieve program insights
  -> generate and execute Gurobi program
  -> compare output objective with y_t
  -> report success or failure
```

The evaluation contract is therefore the same solve path as training, but with
library mutation disabled.

## 6. Full Agent Connectivity

| Stage | Agent | Input | Output | Consumed By |
| --- | --- | --- | --- | --- |
| 1 | Experience Library | Stored insights, taxonomy, task evidence | \(\ell, \tau\) | Retrieval, extraction, evolution |
| 2 | Library Retrieval | \(D_t, M_t\) when available, \(\ell, \tau\), stage | \(I_t^F, I_t^P\) | Formulation Generator, Program Generator |
| 3 | Formulation Generator | \(D_t, I_t^F\) | \(M_t\) | Program Retrieval, Program Generator, Diagnostics |
| 4 | Program Generator | \(D_t, M_t, I_t^P\) | \(C_t, z_t\), execution metadata | Optimality Checker, Diagnostics |
| 5 | Optimality Checker | \(z_t, y_t\), runnable/timeout flags | success status and feedback | Diagnostics, training metrics |
| 6 | Program Diagnostic | \(C_t\), feedback, \(M_t, I_t^P\) | corrected program, code insight labels | Insight Extractor, Library Diagnosis |
| 7 | Formulation Diagnostic | \(D_t, M_t\), feedback, \(G_t\), \(I_t^F\) | \(E_t\), insight labels, missed insights | Insight Extractor, Library Evolution |
| 8 | Self-Exploration | \(D_t, y_t\), failed programs and feedback | proxy gold program \(G_t'\) | Formulation Diagnostic, Insight Extractor |
| 9 | Insight Extractor | failed/corrected artifacts, taxonomy | \(N_t\) | Verification, Merge |
| 10 | Insight Verification | \(N_t\), task, temporary library | verified insights or rejection | Merge, Library Update |
| 11 | Insight Merge | new insights, existing insights | merged canonical insights | Library Update |
| 12 | Library Evolution | \(\ell, \Pi(i)\) | refined \(\ell^{*}\) | Next training iteration, evaluation |

## 7. Why the Architecture Is Agentic

AlphaOPT is not a single prompt that writes solver code. It is an orchestrated
system of specialized agents with explicit contracts:

- The retrieval agent decides which stored knowledge applies.
- The formulation generator builds the mathematical model.
- The program generator writes and executes solver code.
- The optimality checker converts execution into supervision.
- The diagnostic agents explain failures and insight-task interactions.
- The self-exploration agent creates proxy gold programs under answer-only
  supervision.
- The insight extractor turns failures into structured reusable knowledge.
- The verification agent checks whether new knowledge is useful and retrievable.
- The merge agent bounds library growth.
- The evolution agent refines applicability conditions from aggregate evidence.

The autonomy is concentrated where open-ended expert work is required:
selecting relevant modeling lessons, translating text into formulations,
repairing solver programs, abstracting failures into reusable principles, and
adjusting retrieval boundaries.

## 8. Why the Architecture Is Solver-Grounded

AlphaOPT does not rely only on textual plausibility. It introduces execution and
solver feedback at several points:

1. Generated programs must run under Gurobi.
2. Program outputs are compared against known optimal objective values.
3. Self-explored proxy programs are accepted only when they reach the known
   optimum.
4. New insights must solve their source task when replayed.
5. New insights must also be retrievable by the library mechanism.
6. Merged insights are replayed before replacing their source insights.
7. Refined conditions are scored by retrieval behavior over positive, negative,
   and unretrieved task sets.

This makes AlphaOPT a practical experience-learning system for settings where
full expert reasoning traces are unavailable but answer labels and solver
execution are available.

## 9. Robustness Mechanisms

AlphaOPT combines six robustness mechanisms:

**Structured insight representation:** every reusable lesson has taxonomy,
condition, explanation, and example fields rather than free-form memory alone.

**Two-stage retrieval:** taxonomy matching limits the candidate set, and
condition checking prevents inapplicable insights from entering the prompt.

**Solver-grounded task success:** generated programs are judged by executable
objective values, not by whether the text looks mathematically plausible.

**Retrieval-and-success verification:** new insights must both solve the source
task and be retrievable from the library.

**Verified merging:** redundant insights are compressed only when the merged
version preserves replay success.

**Aggregate condition refinement:** insight conditions are revised using
positive, negative, and unretrieved task evidence so they become neither too
broad nor too narrow.

## 10. Final Output Semantics

AlphaOPT's final output is not merely a generated program. A successful
training run produces:

- a refined experience library \(\ell^{*}\),
- an updated taxonomy dictionary \(\tau^{*}\),
- structured insight records with provenance and usage statistics,
- task-interaction profiles \(\Pi(i)\),
- training metrics over task success, verification, merging, and refinement,
- evaluation-time generated formulations \(M_t\),
- evaluation-time generated programs \(C_t\),
- solver objective values \(z_t\),
- success decisions against known answers \(y_t\).

For a downstream task, the accepted result is:

\[
(C_t, z_t)
\quad
\text{such that}
\quad
|z_t - y_t| \leq \epsilon.
\]

For the training process, the accepted durable artifact is:

\[
\ell^{*}
=
\operatorname*{arg\,max}_{\ell \in \mathcal{L}}
\left[
\mathbb{E}_{t \sim \mathcal{T}}
\mathrm{Success}(t \mid \ell)
-
\lambda \Omega(\ell)
\right],
\]

where \(\Omega(\ell)\) represents library complexity, such as size or
redundancy-adjusted size.

## 11. Compact Mental Model

AlphaOPT can be understood as a compiler-like multi-agent architecture with an
evolving memory:

```text
natural language optimization task
  -> retrieve applicable insights
  -> generate mathematical formulation
  -> generate executable Gurobi program
  -> run solver and check answer
  -> diagnose failures
  -> extract structured insights
  -> verify insight usefulness and retrievability
  -> merge redundant insights
  -> refine applicability boundaries
  -> reuse the improved library on future tasks
```

Its central abstraction is the separation between "solve this task" and "learn
when this lesson applies." The solver checks whether a generated program solves
the current task; the library machinery checks whether the lesson extracted
from that attempt can be safely reused. The refinement loop connects them,
allowing AlphaOPT to improve transfer without retraining the underlying model.
