# GridGlyph Trainer Package

The `trainer` package is the neuro-symbolic bridge of the GridGlyph system. It is responsible for fine-tuning small language models (SLMs) to translate visual grid states into verifiable DSL programs. It prioritizes **generalization over memorization** through a zero-footprint, on-the-fly augmentation strategy.

---

## 2. File Responsibilities

### `__init__.py` (The API)
**Primary Responsibility:** Module Exposure.
- **Simplification:** Provides a clean interface to initialize training sessions without exposing internal torch, peft, or transformers complexities.
- **Entry Point:** Exposes `start_fine_tuning()` to trigger the pipeline.

### `generator.py` (The Alchemist)
**Primary Responsibility:** Infinite Data Mutation & Stream Management.
- **On-the-Fly Isomorphism:** Instead of reading static augmented files, it takes the "atomic" seeds and applies random mutations at runtime:
    - **Symbol Swapping:** Replaces digits (0-9) with **Kanji** characters (零, 一, 二...) or alternative symbol sets.
    - **Color Permutation:** Shuffles the mapping of integers to ensure logic is independent of specific "color" values.
- **Entropy Control:** Manages the ratio of original vs. mutated samples to prevent the model from drifting away from standard digit representation while ensuring it remains symbol-agnostic.

### `dataset.py` (The Bridge)
**Primary Responsibility:** Torch Data Piping & Formatting.
- **Iterable Stream:** Implements a `torch.utils.data.IterableDataset` that pulls from the `generator.py` stream. This allows for training on a "virtual" dataset significantly larger than the physical seed file.
- **Tokenization:** Handles the conversion of raw grids and DSL strings into model-specific tokens, optimized for the Qwen architecture.
- **Template Injection:** Wraps samples in the instruction format: 
  `Rule: {dsl_rule} | Input: {input_grid} -> Output: {output_grid}`.

### `model.py` (The Brain Configurator)
**Primary Responsibility:** SLM & LoRA Architecture.
- **Base Model Loading:** Handles the initialization of **Qwen-0.5B** with optional 4/8-bit quantization for frugal, industrial-grade hardware usage.
- **LoRA Integration:** Configures the **PEFT (Parameter-Efficient Fine-Tuning)** adapters.
    - **Targeting:** Focuses on the Linear layers in the Attention blocks (`q_proj`, `v_proj`, `k_proj`, `o_proj`) to capture the mapping between spatial grids and symbolic tokens.
- **Adapter Management:** Handles the merging and saving of weights, keeping the core LLM knowledge frozen while updating the "logical" adapters.

### `train.py` (The Orchestrator)
**Primary Responsibility:** Execution & Rupture Monitoring.
- **Training Loop:** Orchestrates the `transformers.Trainer` or a custom `Accelerate` loop.
- **Hyperparameter Management:** Defines the learning rate schedules, batch sizes, and epoch counts tailored for SLM convergence.
- **Validation Metrics:** Monitors the "Rupture" effect—specifically checking if accuracy remains consistent when the model moves from digit-based puzzles to Kanji-swapped puzzles.


---

## 3. Usage Example

```python
from gridglyph.trainer import start_fine_tuning

# Initialize a LoRA training session on Qwen-0.5B
start_fine_tuning(
    seed_data="training.jsonl",
    model_name="Qwen/Qwen2.5-0.5B",
    use_kanji_augmentation=True,
    lora_rank=16
)