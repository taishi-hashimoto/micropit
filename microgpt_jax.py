"""
A JAX rewrite of the scalar microgpt, keeping the original script layout.
Original scalar version: @karpathy (the code supplied in this conversation).
Whole-sequence attention replaces the sequential KV cache; all else stays local.
"""

import os       # os.path.exists
import random   # random.seed, random.choices, random.gauss, random.shuffle
import jax
import jax.numpy as jnp
random.seed(42) # Let there be order among chaos

# Float64 makes comparison with the original Python float calculations easier.
jax.config.update('jax_enable_x64', True)

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd: JAX replaces the original Value class and backward().

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: jnp.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = state_dict # same dictionary, not a copy; JAX differentiates its arrays directly
print(f"num params: {sum(p.size for p in params.values())}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def softmax(logits):
    return jax.nn.softmax(logits, axis=-1)

def rmsnorm(x):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    scale = (ms + 1e-5) ** -0.5
    return x * scale

@jax.jit
def gpt(state_dict, tokens):
    # tokens: (T,); x: (T, n_embd). Process all positions together.
    pos_id = jnp.arange(len(tokens))
    tok_emb = state_dict['wte'][tokens] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = tok_emb + pos_emb # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[:, hs:hs+head_dim]
            k_h = k[:, hs:hs+head_dim]
            v_h = v[:, hs:hs+head_dim]
            attn_logits = q_h @ k_h.T / head_dim**0.5
            # Causal mask: row t can see columns 0..t, never future tokens.
            attn_logits = jnp.where(pos_id[:, None] >= pos_id[None, :], attn_logits, -jnp.inf)
            attn_weights = softmax(attn_logits)
            head_out = attn_weights @ v_h
            x_attn.append(head_out)
        x_attn = jnp.concatenate(x_attn, axis=-1)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = jax.nn.relu(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head'])
    return logits

# The only added def: put the original forward-to-loss calculation in a function.
def loss_fn(state_dict, tokens):
    n = len(tokens) - 1
    logits = gpt(state_dict, tokens[:-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1) # stable log(softmax(logits))
    losses = -log_probs[jnp.arange(n), tokens[1:]]
    loss = (1 / n) * jnp.sum(losses) # final average loss over the document sequence
    return loss

# Build once, outside the loop. No separate train_step function is needed.
loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = {i: jnp.zeros_like(p) for i, p in params.items()} # first moment buffer
v = {i: jnp.zeros_like(p) for i, p in params.items()} # second moment buffer

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    tokens = jnp.array(tokens[:n+1], dtype=jnp.int32)

    # Forward the sequence to the loss, then backward through all model parameters.
    loss, grads = loss_and_grad(params, tokens)

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in params.items(): # one matrix at a time, instead of one scalar at a time
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grads[i] ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        params[i] = p - lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # No grad reset: loss_and_grad returns fresh gradients on every call.

    print(f"step {step+1:4d} / {num_steps:4d} | loss {float(loss):.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    token_id = BOS
    tokens = [token_id] # recompute the prefix instead of maintaining a KV cache
    sample = []
    for pos_id in range(block_size):
        logits = gpt(state_dict, jnp.array(tokens, dtype=jnp.int32))[-1]
        probs = softmax(logits / temperature)
        token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
        tokens.append(token_id)
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")