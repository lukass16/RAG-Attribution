## Working with Llama Models: Core Concepts and Functions

This guide explains the fundamental concepts and key functions used when working with Meta's Llama model family through the Hugging Face Transformers library. We'll cover model loading, tokenization, generation, and advanced attribution techniques.

## Loading Models and Tokenizers

### AutoTokenizer.from_pretrained()

The `AutoTokenizer` class automatically detects and loads the appropriate tokenizer for a given model.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
```

**Key points:**
- Automatically downloads and caches the tokenizer configuration
- Handles model-specific tokenization rules
- Setting `pad_token = eos_token` is necessary for Llama models since they don't have a dedicated padding token

### AutoModelForCausalLM.from_pretrained()

This function loads a pre-trained causal language model (like Llama) for text generation tasks.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,  # Use half-precision for memory efficiency
    device_map="auto",          # Automatically distribute model across devices
    low_cpu_mem_usage=True      # Optimize memory during loading
)
```

**Parameters explained:**
- `torch_dtype`: Controls numerical precision. `float16` reduces memory usage by 50%
- `device_map="auto"`: Automatically places model layers on available devices (GPU/CPU)
- `low_cpu_mem_usage`: Reduces RAM usage during model initialization

## Tokenization and Encoding

### tokenizer() Function

Converts text strings into numerical token IDs that the model can process.

```python
prompt = "What is the Capital City of Latvia?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
```

**Returns:**
- `input_ids`: Tensor of token IDs representing the input text
- `attention_mask`: Binary mask indicating which tokens are actual inputs vs padding

### tokenizer.convert_ids_to_tokens()

Converts token IDs back to human-readable token strings for inspection.

```python
input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
# Example output: ['What', 'Ġis', 'Ġthe', 'ĠCapital', 'ĠCity', 'Ġof', 'ĠLatvia', '?']
```

Note: The `Ġ` symbol represents a space before the token in byte-pair encoding.

### tokenizer.decode()

Converts token IDs back to a readable text string.

```python
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Parameters:**
- `skip_special_tokens=True`: Removes special tokens like `<s>`, `</s>`, `<pad>` from output

## Text Generation

### model.generate()

The primary function for generating text completions with language models.

```python
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
```

**Key parameters:**
- `max_new_tokens`: Maximum number of tokens to generate
- `do_sample=False`: Use greedy decoding (always pick highest probability token)
- `do_sample=True`: Enable sampling for more diverse outputs
- `pad_token_id`: Specifies which token ID to use for padding

**Advanced generation parameters:**
- `temperature`: Controls randomness (lower = more deterministic)
- `top_k`: Only sample from top k most likely tokens
- `top_p`: Nucleus sampling - sample from smallest set with cumulative probability ≥ p

## Understanding Model Embeddings

### model.get_input_embeddings()

Returns the embedding layer that converts token IDs into dense vector representations.

```python
embeddings_layer = model.get_input_embeddings()
token_embeddings = embeddings_layer(input_ids)
# Shape: (batch_size, sequence_length, embedding_dim)
```

**What are embeddings?**
Embeddings are learned dense vector representations where each token is represented as a high-dimensional vector (typically 2048+ dimensions for Llama models). These vectors capture semantic relationships between tokens.

## Model Forward Pass

### model() and model(inputs_embeds=...)

The model can be called directly for a forward pass through the network.

```python
# Standard forward pass with token IDs
outputs = model(input_ids)

# Forward pass with custom embeddings
outputs = model(inputs_embeds=token_embeddings)
```

**Returns an object containing:**
- `logits`: Raw prediction scores for each token in vocabulary (shape: batch_size × seq_len × vocab_size)
- `hidden_states`: Intermediate layer outputs (if `output_hidden_states=True`)
- `attentions`: Attention weights (if `output_attentions=True`)

### Understanding Logits

Logits are unnormalized prediction scores. To get probabilities:

```python
import torch.nn.functional as F

logits = outputs.logits[:, -1, :]  # Get logits for last position
probabilities = F.softmax(logits, dim=-1)  # Convert to probabilities
predicted_token_id = logits.argmax(dim=-1)  # Get most likely token
```

## Gradient-Based Attribution

### Computing Gradients with respect to Embeddings

This technique reveals which input tokens most influence the model's predictions.

```python
def gradient_based_attribution(model, input_ids, target_token_idx=-1):
    # Get embeddings and enable gradient tracking
    embeddings = model.get_input_embeddings()
    token_embeddings = embeddings(input_ids)
    token_embeddings.requires_grad_(True)
    
    # Forward pass
    outputs = model(inputs_embeds=token_embeddings)
    logits = outputs.logits
    
    # Get predicted token and its score
    target_logits = logits[0, target_token_idx]
    predicted_token_id = target_logits.argmax(dim=-1)
    target_score = target_logits[predicted_token_id]
    
    # Compute gradients
    target_score.backward()
    gradients = token_embeddings.grad
    
    # Attribution score = L2 norm of gradients
    attribution_scores = gradients.norm(dim=-1).squeeze().cpu().detach().numpy()
    
    return attribution_scores, predicted_token_id.item()
```

**Key concepts:**
- `requires_grad_(True)`: Enables gradient computation for the embeddings
- `backward()`: Backpropagates to compute gradients
- `gradients.norm(dim=-1)`: L2 norm captures gradient magnitude across embedding dimensions
- Higher gradient magnitude = token has more influence on prediction

## Attention Weight Analysis

### Extracting Attention Weights

Attention mechanisms show which tokens the model "pays attention to" at each layer.

```python
def extract_attention_weights(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Get attention from all layers
    attentions = outputs.attentions
    # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    
    # Stack and average across layers and heads
    all_attention = torch.stack(attentions)
    avg_attention = all_attention.mean(dim=(0, 2))
    
    # Get attention from last token to all previous tokens
    last_token_attention = avg_attention[0, -1, :].cpu().numpy()
    
    return last_token_attention, all_attention
```

**Understanding attention shapes:**
- Each layer has multiple attention heads (typically 32 for Llama models)
- Attention matrix is (seq_len × seq_len) showing relationships between all token pairs
- `attentions[layer][batch][head][query_token][key_token]` = attention weight

**Averaging strategies:**
- Average across heads: Get overall attention pattern per layer
- Average across layers: Get global attention pattern
- Focus on last token: See what model attends to when making next prediction

## Integrated Gradients

### Advanced Attribution with Captum

Integrated Gradients provides theoretically grounded attribution by computing gradients along a path from baseline to input.

```python
from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, input_ids, baseline_type='zero'):
    embeddings_layer = model.get_input_embeddings()
    input_embeddings = embeddings_layer(input_ids)
    
    # Create baseline (reference point)
    if baseline_type == 'zero':
        baseline_embeddings = torch.zeros_like(input_embeddings)
    else:
        pad_token_id = tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_token_id)
        baseline_embeddings = embeddings_layer(baseline_ids)
    
    # Get predicted token
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_token_id = outputs.logits[0, -1].argmax().item()
    
    # Initialize and compute
    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeddings,
        target=predicted_token_id,
        n_steps=50,
        internal_batch_size=1
    )
    
    # Compute attribution scores
    attribution_scores = attributions.norm(dim=-1).squeeze().cpu().detach().numpy()
    
    return attribution_scores, predicted_token_id
```

**Key parameters:**
- `baselines`: Reference input to compare against (typically zeros or padding tokens)
- `target`: Which output class/token to compute attributions for
- `n_steps`: Number of steps in the integral approximation (more = more accurate but slower)
- `internal_batch_size`: Batch size for internal computation

**Why use Integrated Gradients?**
- Satisfies completeness: Attributions sum to difference between output at input and baseline
- More stable than simple gradients
- Less sensitive to gradient saturation

## Model Forward Function for Captum

```python
def model_forward(embeddings, attention_mask=None):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits[:, -1, :]  # Return logits for last position
```

This wrapper function is required by Captum's attribution methods. It takes embeddings as input and returns the logits you want to attribute.

## Working with Model Layers

### Accessing Internal Layers

Llama models have a hierarchical structure:

```python
# Number of transformer layers
num_layers = len(model.model.layers)

# Access specific layer
first_layer = model.model.layers[0]

# Components within each layer:
# - self_attn: Self-attention mechanism
# - mlp: Feed-forward network
# - input_layernorm: Layer normalization before attention
# - post_attention_layernorm: Layer normalization after attention
```

## Memory Management and Optimization

### torch.no_grad() Context Manager

Disables gradient computation to save memory during inference.

```python
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=20)
```

**When to use:**
- During inference/generation (not training)
- When you don't need to compute gradients
- Significantly reduces memory usage

### model.eval() vs model.train()

```python
model.eval()  # Set to evaluation mode
```

**Effects of eval mode:**
- Disables dropout (makes outputs deterministic)
- Changes batch normalization behavior
- Essential for consistent inference results

## Tensor Operations and Device Management

### Moving Tensors to Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

**Key functions:**
- `.to(device)`: Moves tensor to specified device (CPU/GPU)
- `.cpu()`: Moves tensor to CPU
- `.cuda()`: Moves tensor to GPU

### Common Tensor Operations

```python
# Get tensor shape
shape = tensor.shape  # or tensor.size()

# Indexing: Get last position
last_token_logits = logits[:, -1, :]

# Squeeze: Remove dimensions of size 1
squeezed = tensor.squeeze()

# Stack: Combine tensors along new dimension
stacked = torch.stack([tensor1, tensor2])

# Mean: Average across dimensions
averaged = tensor.mean(dim=0)  # Average along dimension 0
```

## Attribution Visualization Best Practices

### Normalizing Scores for Comparison

```python
def normalize_scores(scores):
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

grad_scores_norm = normalize_scores(grad_scores)
```

This rescales attribution scores to [0, 1] range, making different methods comparable.

### Creating Attribution Matrices

```python
attribution_matrix = np.vstack([
    grad_scores_norm,
    attention_scores_norm,
    ig_scores_norm
])
```

Stacking normalized scores allows side-by-side comparison of different attribution methods.

## Common Patterns and Tips

### Pattern 1: Extracting Next Token Prediction

```python
with torch.no_grad():
    outputs = model(input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    predicted_token_id = next_token_logits.argmax()
    predicted_token = tokenizer.decode([predicted_token_id])
```

### Pattern 2: Getting Top-K Predictions

```python
top_k = 5
top_logits, top_indices = next_token_logits.topk(top_k)
top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
```

### Pattern 3: Batch Processing

```python
prompts = ["Question 1?", "Question 2?", "Question 3?"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=20)
```

## Model Configuration and Properties

### Accessing Model Configuration

```python
config = model.config

print(f"Vocabulary size: {config.vocab_size}")
print(f"Hidden size: {config.hidden_size}")
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Number of attention heads: {config.num_attention_heads}")
print(f"Max position embeddings: {config.max_position_embeddings}")
```

### Getting Model Size

```python
num_parameters = model.num_parameters()
num_parameters_millions = num_parameters / 1e6
print(f"Model has {num_parameters_millions:.2f}M parameters")
```

## Debugging and Inspection

### Checking Shapes

Always verify tensor shapes match expectations:

```python
print(f"Input IDs shape: {input_ids.shape}")  # [batch_size, seq_len]
print(f"Embeddings shape: {embeddings.shape}")  # [batch_size, seq_len, hidden_dim]
print(f"Logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
```

### Inspecting Token Sequences

```python
# See both IDs and tokens
for i, (token_id, token) in enumerate(zip(input_ids[0], input_tokens)):
    print(f"Position {i}: ID={token_id}, Token='{token}'")
```

## Performance Considerations

### Memory-Efficient Generation

For long sequences, generate in chunks or use techniques like:

```python
# Use gradient checkpointing during training
model.gradient_checkpointing_enable()

# Quantize model to 8-bit for inference
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### Batch Size Optimization

Start with small batch sizes and increase until you hit memory limits:

```python
# For attribution methods that process multiple inputs
internal_batch_size = 1  # Increase if memory allows
```

## Summary of Key Functions

**Model Loading:**
- `AutoTokenizer.from_pretrained()`: Load tokenizer
- `AutoModelForCausalLM.from_pretrained()`: Load language model

**Tokenization:**
- `tokenizer()`: Convert text to token IDs
- `tokenizer.decode()`: Convert token IDs to text
- `tokenizer.convert_ids_to_tokens()`: Get individual token strings

**Generation:**
- `model.generate()`: Generate text completions
- `model()`: Forward pass through model
- `model.get_input_embeddings()`: Access embedding layer

**Attribution:**
- `tensor.backward()`: Compute gradients via backpropagation
- `model(output_attentions=True)`: Extract attention weights
- `IntegratedGradients.attribute()`: Compute integrated gradients

**Tensor Operations:**
- `.to(device)`: Move tensors between CPU/GPU
- `.requires_grad_()`: Enable gradient tracking
- `.norm()`: Compute vector norms
- `.mean()`, `.argmax()`, `.topk()`: Common aggregation operations

This guide covers the essential concepts and functions for working with Llama models, from basic text generation to advanced attribution techniques for understanding model behavior.

