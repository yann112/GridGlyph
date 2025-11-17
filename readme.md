# GridGlyph

## A Symbolic Reasoning Engine for Solving Visual Logic Puzzles

GridGlyph is a hybrid **symbolic + AI** system designed to solve grid-based visual reasoning puzzles (like those in the Abstraction and Reasoning Corpus — ARC). It transforms numeric grids into abstract symbolic representations and learns **deterministic transformation rules** using a **latent embedding space**.

Unlike traditional approaches, GridGlyph ensures:

* Multiple puzzle instances of the same underlying rule converge in the **latent embedding space**
* DSL rules are deterministic and interpretable
* Novel combinations of atomic operations are possible without hallucination

---

## 🧩 Core Concepts

### 1. GridGlyph (Project Name)

> The full pipeline that solves visual pattern puzzles using symbolic abstraction and embeddings.

GridGlyph teaches AI to see transformations **structurally**:

* Not by raw numeric values
* But by **object counts, patterns, repetition, mirroring, rotation**
* Using embeddings → DSL predictor → execution → scoring

---

### 2. Embedding-Based Reasoning & Rule Generalization

> Encode puzzles into a latent space capturing structural features, **forcing multiple representations of the same rule to converge**.

* **Multiple views per rule**:
  Every puzzle rule is represented by **all available input/output examples**, possibly with symbolic perturbations (shuffled numbers, reordered objects, etc.) to create diverse variants.

* **Convergent embedding**:
  Feeding all these variants into the embedding encoder forces **latent vectors of the same rule to align**, independent of numeric ordering.

* **Single-block model training**:
  A compact model predicts the **DSL sequence** directly from the embedding.
  Since embeddings converge, the model produces a **unique, deterministic DSL output** for each underlying transformation.

**Benefits**:

* Deterministic rule output
* No hallucination
* Supports novel recombinations of atomic operations
* Simplified, compact model training

---

### 3. sigil (Transformation Logic)

> Transformation rules expressed as executable logic (DSL).

A `sigil` can be:

* A Python function generated from the model
* Or a structured DSL sequence representing atomic operations

**Example Python Sigil**:

```python
def transform(grid):
    row1 = grid[0] * 3
    row2 = grid[1] * 3
    return [
        row1,
        row2,
        row1[::-1],
        row2[::-1]
    ]
```

**Example DSL Sigil**:

```
T = [repeat_horizontal(R0, 3), repeat_horizontal(R1, 3),
     mirror_rows(R0), mirror_rows(R1)]
```

**Why sigils**:

* Encodes transformations in a deterministic, interpretable way
* Can be executed directly for scoring
* Supports recombination and extension with atomic DSL operations

---

## 🔁 Pipeline Overview

1. **Input Puzzle**: Numeric input/output grid pairs
2. **Create multiple variants**: Generate several puzzle instances for the same rule (object shuffle, alternate symbols, numeric remapping)
3. **Embedding Encoder**: Map grids to latent space, forcing same-rule embeddings to cluster
4. **DSL Prediction Model**: Train a compact model to output DSL sequences from embeddings
5. **Execute Sigil**: Run predicted transformation on input grids
6. **Score**: Compare predicted output vs expected results
7. **Optional Feedback Loop**: Retry or augment if outputs are inconsistent

---

## 🧠 Why This Approach Works

* **Embedding alignment** ensures deterministic rule outputs
* **Multiple puzzle variants** improve generalization
* **DSL-based execution** eliminates hallucination
* **Atomic operations + recombination** allows solving unseen puzzles

---

## ⚡ Advantages

* Deterministic, interpretable transformation rules
* Can handle many variations of the same underlying rule
* Compact models, less compute than full LLMs
* Supports recombination of atomic transformations for unseen puzzles
* Embeddings encourage generalization and rule alignment

---

## 🛠️ Development Notes

* Start with **atomic rules**, validate manually
* Generate **multiple variants per rule** for embedding convergence
* Train embeddings and DSL predictor **jointly**
* Track rule execution and scores to ensure convergence

---
