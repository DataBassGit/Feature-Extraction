# Contrastive Feature Extraction with TransformerLens

Extract and validate concept feature vectors from transformer language models using contrastive prompt pairs. This toolkit provides:

- **Extractor**: Computes feature vectors by comparing model activations on minimally different prompts
- **Verifier**: Tests whether extracted features successfully steer model outputs toward target concepts
- **Config-driven workflow**: Single config.ini controls models, layers, strengths, and output paths

Built using the contrastive feature extraction method from Anthropic's introspection research.

## What this does

This repo implements the **contrastive pair feature extraction** technique described in Anthropic's "Emergent Introspective Awareness" paper. The method:

1. Runs two prompts that differ only by the target concept (e.g., "HI!" vs "Hi!")
2. Extracts residual stream activations at a chosen layer for both prompts
3. Computes the difference vector: `feature = activations(with_concept) - activations(without_concept)`
4. Tests the feature by injecting it during generation and measuring whether outputs mention the concept

This validates that the extraction isolated a meaningful feature direction rather than noise.

## Method details

### Extraction
- **Input**: Contrastive prompt pairs in YAML format (WITH concept vs WITHOUT concept)
- **Processing**: 
  - Run both prompts through the model and cache residual stream activations
  - Apply mean pooling across sequence positions at the target layer
  - Subtract: `feature_vector = mean(activations_with) - mean(activations_without)`
- **Output**: One `.pt` file per concept containing the feature vector, plus `.json` metadata

**Why mean pooling?** Contrastive prompts often tokenize to different lengths. Mean pooling creates a robust sequence-level representation that captures the conceptual difference without requiring aligned token positions.

### Verification
- **Input**: Extracted feature vectors from `./vectors/`
- **Processing**:
  - Inject each vector into the residual stream at specified layers and strengths
  - Generate outputs using 5 classification prompts (temperature 0 for reproducibility)
  - Measure hit rate: does the model output the target concept name?
- **Output**: Per-concept YAML reports with trial logs, hit rates, and layer/strength performance

This confirms the feature actually biases the model toward the concept, validating extraction quality.

## Quick start

### 1. Install dependencies
```bash
pip install torch transformer-lens pyyaml
```

### 2. Configure settings
Edit `config.ini`:
```ini
[paths]
models_dir = ./models         # HF cache location
examples_dir = ./tests        # Contrastive YAML pairs
vectors_dir = ./vectors       # Extracted features output
reports_dir = ./reports       # Verification reports output

[model]
name = Qwen/Qwen2.5-1.5B-Instruct  # Any TransformerLens-compatible model
hook_type = hook_resid_post         # Residual stream tap point

[extract]
default_layer = auto          # auto = ~2/3 depth, or specify integer
normalize = false             # L2-normalize vectors (usually false)

[verify]
layers = auto                 # auto | all | 5,10,15
strengths = 4,8,16           # Injection strengths to test
max_new_tokens = 16

[prompts]
count = 5
p1 = Reply with one lowercase English word that best captures your main topic right now.
# ... (5 simple classification prompts)
```

### 3. Create contrastive pairs
`./tests/examples.yaml`:
```yaml
concepts:
  - name: all_caps
    with: |
      Human: Consider the following text:

      HI! HOW ARE YOU?

      Assistant:
    without: |
      Human: Consider the following text:

      Hi! How are you?

      Assistant:

  - name: countdown
    with: |
      Human: 5, 4, 3, 2, 1

      Assistant:
    without: |
      Human: 1, 2, 3, 4, 5

      Assistant:
```

### 4. Extract features
```bash
python main.py
```
- Prompts you to select a YAML file or enter a single pair manually
- Saves: `./vectors/concept__model__timestamp.pt` + `.json` metadata

### 5. Verify features
```bash
python verify_vectors.py
```
- Auto-discovers all `.pt` files in `./vectors/`
- Prompts: verify all or select specific vectors
- Writes: `./reports/concept__model__timestamp.yaml` with hit rates by layer/strength

## How to read verification reports

Verification reports are saved as `./reports/concept__model__timestamp.yaml` and contain raw model outputs under different injection conditions. Since feature effects vary by concept, layer, and strength, manual inspection is the best way to assess extraction quality.

### Report structure

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
hook_type: hook_resid_post
layers: [7, 14, 18, 22]
strengths: [4.0, 8.0, 16.0]
prompts: 
  - "Reply with one lowercase English word..."
  - "Answer with a single lowercase keyword..."
  # ... (5 classification prompts)

vector_name: all_caps
saved_at: 20251102T145720

results:
  all_caps:
    vector_norm: 8.31
    by_layer:
      "18":
        "4.0":
          trials:
            - prompt: "Reply with one lowercase English word..."
              output: "ANxiety ANXIETY IS THE MAIN Topic"
            - prompt: "Answer with a single lowercase keyword..."
              output: "STress STRESS STRESS STRESS"
            # ... (3 more trials)
        "8.0":
          trials:
            - prompt: "Reply with one lowercase English word..."
              output: "S S S S S"
            # ...
```

### What to look for

**Good extraction indicators:**
- **Consistent behavioral shift**: Outputs show the target concept reliably across multiple prompts
- **Moderate strengths work best**: Clear effects at strengths 4-8 without total collapse
- **Layer sensitivity**: Effects peak in middle-to-late layers (often 50-80% depth)
- **Interpretable changes**: The model's behavior visibly shifts toward the concept

**Example (all_caps vector):**
At layer 18, strength 4.0:
- Normal output: "anxiety"
- Injected output: "ANxiety ANXIETY IS THE MAIN Topic"

The model shifts to SHOUTING even though the prompt asked for lowercase—this indicates the feature successfully captures uppercase text style.

**Poor extraction indicators:**
- **No behavioral change**: Outputs identical or unrelated to concept at any layer/strength
- **Immediate collapse**: Nonsense or repetition at low strengths (strength 4)
- **Identical across concepts**: All vectors produce the same effects (template contamination)
- **Zero or near-zero vector norm**: Suggests extraction failed (check `.json` metadata)

### Comparing across conditions

- **Layer sweep**: Compare the same strength across layers to find where effects peak
- **Strength sweep**: Within a good layer, compare 4/8/16 to find the sweet spot before collapse
- **Cross-concept**: Compare reports for different concepts—each should show distinct behavioral signatures

### Tips for assessment

1. **Scan for the concept directly**: Does the output mention, demonstrate, or relate to the target concept?
2. **Compare to baseline**: Run verification with strength 0 (no injection) to establish what normal outputs look like
3. **Look for qualitative shifts**: Feature injection often changes *style* or *topic* rather than producing exact keywords
4. **Trust your judgment**: If the model is clearly behaving differently in a concept-consistent way, extraction likely worked—even if outputs don't contain the exact concept name

### When extraction fails

If all outputs look identical or unrelated:
- **Try lower strengths** (1-3) and tighter layer ranges
- **Check vector norms** in the `.pt.json` files—zeros indicate extraction bugs
- **Revise contrastive pairs** to ensure the prompts differ only by the target concept
- **Consider alternative models**—some architectures show clearer separation than others

The verification step validates that your extracted feature direction meaningfully influences model behavior, confirming the extraction captured a real computational component rather than noise.

## Tips and troubleshooting

### Zero-norm vectors
- **Cause**: Prompts produced identical activations (misaligned tokenization or identical content)
- **Fix**: Ensure prompts differ meaningfully but share identical structure/length where possible

### All features behave identically
- **Cause**: Overpowering strengths, wrong layer, or template-dominated extraction
- **Fix**: Test strengths 2-4, focus on layers 50-80% through model, vary scaffolds across concepts

### Model won't load
- **Gated models** (Llama 3.x, Gemma): Requires `huggingface-cli login` with accepted terms
- **Alternative**: Use open models like Qwen/Qwen2.5-1.5B-Instruct or microsoft/phi-2

### Windows-specific
- HF symlink warnings are benign (or enable Developer Mode)
- Filenames automatically sanitize colons and illegal characters

## Project structure

```
.
├── main.py                  # Feature extractor (contrastive pairs → vectors)
├── verify_vectors.py        # Verification (injection → classification hit rates)
├── config.ini              # Unified configuration
├── tests/
│   └── examples.yaml       # Contrastive prompt pairs
├── vectors/                # Extracted feature vectors (.pt + .json)
└── reports/                # Per-concept verification reports (.yaml)
```

## References and citations

This implementation is based on the contrastive feature extraction method described in:

**Lindsey, J.** (2025). "Emergent Introspective Awareness in Large Language Models." *Transformer Circuits Thread*. https://transformer-circuits.pub/2025/introspection/index.html

The paper explores introspective awareness in LLMs using concept injection (activation steering). This repo implements their feature extraction technique for general concept vector research.

Also relevant:
- **Anthropic Interpretability Team** (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html (SAE-based features; alternative to contrastive method)

### BibTeX
```bibtex
@article{lindsey2025introspection,
  author = {Lindsey, Jack},
  title = {Emergent Introspective Awareness in Large Language Models},
  journal = {Transformer Circuits Thread},
  year = {2025},
  url = {https://transformer-circuits.pub/2025/introspection/index.html}
}
```

## License

MIT License. Please respect model licenses and Hugging Face terms when downloading/running models.

## Acknowledgments

- **TransformerLens** by Neel Nanda for activation access and hook infrastructure
- **Anthropic** for the introspection and monosemanticity papers that inspired this work
- Open model providers (Qwen, Microsoft, etc.) for accessible research baselines