
---

# 🧠 Prompt Engineering Cheat Sheet (Synthetic & Actionable)

A concise, practical reference to help design **LLM prompts** that:
- Encourage symbolic reasoning
- Reduce format bias
- Leverage glyph abstraction and sigil logic
- Improve performance on ARC-style puzzles

---

## 🔁 Core Prompt Strategy Template

```plaintext
You are given input and output grids. Study how elements move between positions.

Describe the transformation using only these sigils:

🜌 = mirror_rows  
🜋 = mirror_cols  
🜐 = shift_down  
🜑 = shift_right  
🜊 = tile_horizontal  
🜉 = repeat_first  
🜓 = diagonal_to_row  

Do not use natural language descriptions like "swap rows" or "shift down". Use only sigils.

Apply the same rule to this new input:
[TEST_INPUT_GRID]

Write your transformed output below.
```

---

## 🧪 Prompt Engineering Techniques (Cheat List)

| Prompt Strategy | How to Apply It | Goal |
|----------------|------------------|------|
| **Rule Description First** | Ask model to describe transformation before applying it | Force abstraction |
| **Sigil-Only Instructions** | Restrict model to predefined sigil set | Reduce ambiguity |
| **Few-Shot with Glyph Variation** | Show same puzzle with different glyphs | Prevent symbol anchoring |
| **Contrastive Examples** | Give one incorrect output | Build diagnostic reasoning |
| **Step-by-Step Position Mapping** | Trace Input[i][j] → Output[i][j] | Improve spatial understanding |
| **Chain-of-Thought Reasoning** | Ask model to break down logic step-by-step | Encourage deeper analysis |
| **Inverse Task Prompting** | Give output and ask for input under known sigil | Test rule comprehension |
| **Feature Hints** | Count symbols, compare row/column stats | Guide attention |
| **Transformation Chaining** | Combine multiple sigils in sequence | Teach compositionality |
| **Rotation/Mirror Variants** | Show rotated/mirrored version | Test spatial logic |
| **Metacognitive Feedback** | Ask model to reflect on failed attempt | Improve error correction |
| **Diagram Requests** | Ask for text-based element movement diagrams | Visualize logic |

---

## 🧩 Glyph Remapping Strategy (Data Augmentation)

Use these when prompting or training:

| Glyph Set | Example Grid | Purpose |
|-----------|--------------|---------|
| Katakana  | ツ ア ヤ キ ツ | Human-readable abstraction |
| Emoji     | ⚡ 💥 🌀 ✅ ⚠️ | Strong visual pattern recognition |
| Dice Dots | ⚀ ⚁ ⚂ ⚃ ⚄ | No numeric assumptions |
| Stroke Order | ㇀ ㇁ ㇂ ㇃ | Sequential structure awareness |
| Custom Glyphs | 🜀 🜁 🜂 🜃 🜄 | Neutral representation |
| Box Drawing | ┌ ─ ┐ │ ┘ | Spatial relationship emphasis |

> 💡 Tip: Change glyph sets across examples to force generalization.

---

## 🧰 Sigil Operation Reference

| Sigil | Meaning | Example Use Case |
|-------|---------|------------------|
| 🜀 | Identity | No change needed |
| 🜌 | Mirror Rows Horizontally | Flip grid vertically |
| 🜋 | Mirror Columns Vertically | Flip within-row values |
| 🜐 | Shift Down | Wrap bottom row to top |
| 🜑 | Shift Right | Wrap last column to left |
| 🜊 | Tile Horizontally | Repeat row content |
| 🜉 | Repeat First Element | Fill row with first value |
| 🜓 | Diagonal to Row | Map diagonals into rows |
| 🜔 | Rotate Clockwise | Rotate entire grid 90° |
| 🜒 | Append Input Below | Add original input as new rows |
| 🜕 | Color Inversion | Swap foreground/background colors (for color puzzles) |

> 💡 You can define more based on common transformations you observe.

---

## 🔄 Prompt Pattern Templates (Copy-Paste Ready)

### 🎯 1. Basic Rule Discovery

```plaintext
Study the transformation carefully.

Input:
ツ ア  
ヤ キ  

Output:
ア ツ  
キ ヤ  

Apply the same rule to this new input:
モ オ  
ネ コ  

Describe the transformation using only these sigils: 🜌 🜋 🜐 🜑 🜊 🜉 🜓

Answer:
[🜌]
```

---

### 🧩 2. Few-Shot with Glyph Variation

```plaintext
Same transformation, different symbols:

Example A:
✿ ★  
• ■  

→  
★ ✿  
■ •  

Example B:
ツ ア  
ヤ キ  

→  
ア ツ  
キ ヤ  

Now apply same rule to test input:
モ オ  
ネ コ  
```

---

### 🧱 3. Contrastive Reasoning

```plaintext
One of these outputs was NOT generated using the correct rule.

Option A:
ツ ア  
ア ヤ  

Option B:
ヤ ア  
ツ キ  

Which is incorrect? Explain why using sigils.
```

---

### 🧮 4. Step-by-Step Mapping

```plaintext
Trace how each element moves:

Input[0][0] = ツ → Output[1][0] = ヤ  
Input[0][1] = ア → Output[1][1] = キ  
Input[1][0] = ヤ → Output[0][0] = ツ  
Input[1][1] = キ → Output[0][1] = ア  

What sigil(s) represent this transformation?

Available: 🜌 🜋 🜐 🜑 🜊 🜉 🜓
```

---

### 🔄 5. Transformation Chaining

```plaintext
Apply this sigil sequence: 🜌🜊  
(mirror then tile horizontally)

Input:
ツ ア  
ヤ キ  
```

---

### 🧭 6. Metacognitive Reflection

```plaintext
Your first guess was 🜐 (shift down), but that doesn't match the output.

Try again. Could it be 🜌 (mirror rows)? Or something else?

Explain why your new guess should work.
```

---

## 🛠️ Bonus: Prompt Optimization Tips

| Tip | Why It Helps |
|-----|--------------|
| Keep prompts short and focused | Avoids cognitive overload |
| Use consistent sigil set | Reduces learning complexity |
| Vary glyph sets per example | Forces generalization |
| Always show input → output | Teaches mapping behavior |
| Include failure cases | Improves error detection |
| Ask for rule before output | Prevents guessing |
| Use small grids first | Easier debugging |
| Gradually increase difficulty | Builds skill incrementally |
| Track successful patterns | Reuse effective structures |
| Name sigil sequences | E.g., `🜓` = diagonal_to_row |

---

## 📋 Summary Table: Prompt Ingredients

| Ingredient | Effect |
|------------|--------|
| Glyph variation | Prevents memorization |
| Sigil grammar | Formalizes transformation rules |
| CoT / position tracing | Encourages deep analysis |
| Contrastive examples | Builds diagnostic reasoning |
| Chain-of-thought | Guides logical breakdown |
| Mini-language instructions | Standardizes transformation language |
| Diagram requests | Supports spatial reasoning |
| Inverse tasks | Tests true understanding |
| Rotation/mirror variants | Validates spatial logic |
| Feature hints | Directs attention to stable properties |

---

## 🧩 Final Thought

This cheat sheet gives you a **repeatable, scalable framework** for designing **high-quality prompts** that guide LLMs toward **symbolic, structured reasoning** — not just pattern matching.

You’re not just asking questions — you're **teaching the model how to think**.

---

Would you like me to:
- Export this as a printable PDF or Markdown file?
- Generate a **prompt template generator** (Python-ready)?
- Create a **cheat sheet poster** for quick reference?
- Build a **prompt optimization tool** that suggests better versions?

Let me know and I’ll help you package your knowledge into a powerful **prompt engineering toolkit**. 💡
