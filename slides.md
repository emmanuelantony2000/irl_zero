<!-- .slide: class="title-slide" -->

# AI on Chip &<br /><span class="highlight2">Diffusion Models</span>

<p style="color: #888; font-size: 0.8em; margin-top: 30px">
Custom Silicon for AI Inference & The Future of Token Generation
</p>

---

<!-- .slide: class="section-divider" -->

## Part I

<p class="subtitle">AI on Chip</p>
<p style="font-size: 0.7em; color: #555">
Custom silicon &bull; Taalas
</p>

---

## The Problem with Current AI Inference

- GPUs are <span class="highlight">general-purpose</span> &mdash; not optimized just for inference
- The <span class="highlight">memory wall</span>: data movement between memory and compute is the bottleneck

---

## How Inference Works at the Chip Level

### Traditional GPU/TPU Path

<div class="flow-diagram">
<div class="flow-step">Model<br />Weights</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Load to<br />Compute Unit</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Matrix<br />Multiply</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Activation<br />Functions</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Next<br />Token</div>
</div>

- Each token requires <span class="highlight">loading billions of parameters</span> from memory to compute cores
- Bottleneck is <span class="highlight2">memory bandwidth</span>, not compute
- GPUs spend most of their time <span class="highlight">waiting for data</span>, not computing

---

## A Fundamentally Different Approach (by Taalas)

- <span class="highlight">Total Specialization</span> &mdash; custom silicon optimized for a *specific* AI model, not general-purpose
- <span class="highlight">Unified Memory-Compute</span> &mdash; storage and compute on a single chip at DRAM-level density, eliminating the memory wall

---

## GPU vs TPU vs AI-on-Chip

<table class="comparison-table">
<thead>
<tr>
<th></th>
<th>GPU (NVIDIA)</th>
<th>TPU (Google)</th>
<th>Taalas HC1</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Design</strong></td>
<td>General-purpose parallel</td>
<td>Matrix-op specialized</td>
<td>Model-specific silicon</td>
</tr>
<tr>
<td><strong>Memory</strong></td>
<td>HBM (separate)</td>
<td>HBM (separate)</td>
<td>Unified on-chip</td>
</tr>
<tr>
<td><strong>Bottleneck</strong></td>
<td>Memory bandwidth</td>
<td>Memory bandwidth</td>
<td>Compute (memory is free)</td>
</tr>
<tr>
<td><strong>Flexibility</strong></td>
<td>Any model</td>
<td>Any TF/JAX model</td>
<td>One model per chip</td>
</tr>
<tr>
<td><strong>Use Case</strong></td>
<td>Training + Inference</td>
<td>Training + Inference</td>
<td>Inference only</td>
</tr>
</tbody>
</table>

---

## Taalas HC1 Performance

<p style="font-size: 0.8em; color: #aaa">
Benchmarked on Llama 3.1 8B
</p>

<span class="number">17,000</span>
<span class="label">tokens/sec per user</span>

<p class="small">
NVIDIA H200 comparison: ~1,700 tokens/sec
</p>

---

## AI-on-Chip: Pros & Cons

<div class="two-col">
<div class="col-box pros">
<h3>Advantages</h3>
<ul>
<li>Extreme inference speed (10x faster)</li>
<li>Dramatically lower cost (20x cheaper)</li>
<li>10x less power consumption</li>
<li>Near-zero marginal inference cost</li>
<li>Eliminates the memory wall entirely</li>
</ul>
</div>
<div class="col-box cons">
<h3>Disadvantages</h3>
<ul>
<li>One model per chip &mdash; no flexibility</li>
<li>New model requires new silicon</li>
<li>Cannot be used for training</li>
<li>2-month fabrication cycle per model</li>
<li>Not viable for rapidly evolving model families</li>
</ul>
</div>
</div>

---

<!-- .slide: class="section-divider" -->

## Part II

<p class="subtitle">Diffusion Models for LLMs</p>
<p style="font-size: 0.7em; color: #555">
Parallel token generation
</p>

---

## How Transformer LLMs Think

### Autoregressive Generation

<div class="flow-diagram">
<div class="flow-step">Prompt</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Token 1</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Token 2</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Token 3</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">...</div>
</div>

- Generates <span class="highlight">one token at a time</span>, left-to-right
- Each token depends on *all previous tokens* &mdash; inherently sequential
- <span class="highlight2">O(n)</span> forward passes for n tokens &mdash; speed scales linearly
- "Thinking" (chain-of-thought) just means generating more tokens sequentially
- KV-cache grows with context &mdash; memory pressure at long sequences

---

## What Are Diffusion Models?

### The Core Idea: Denoising

<div class="flow-diagram">
<div class="flow-step" style="background: rgba(255,70,100,0.15); border-color: rgba(255,70,100,0.3)">
Pure<br />Noise
</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step" style="background: rgba(255,130,80,0.12); border-color: rgba(255,130,80,0.3)">
Mostly<br />Noise
</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step" style="background: rgba(255,200,50,0.1); border-color: rgba(255,200,50,0.3)">
Some<br />Structure
</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step" style="background: rgba(100,255,100,0.1); border-color: rgba(100,255,100,0.3)">
Almost<br />Clear
</div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step" style="background: rgba(0,212,255,0.15); border-color: rgba(0,212,255,0.3)">
Clean<br />Output
</div>
</div>

- Start with <span class="highlight2">random noise</span>, iteratively refine toward the target
- Model learns the <span class="highlight">reverse</span> of a noise-adding process
- Each step refines **all positions simultaneously**
- Originally designed for images &mdash; now being adapted for <span class="highlight">text generation</span>

---

## Diffusion Models for Text Generation

### Parallel Token Processing

<p style="font-size: 0.8em; color: #aaa; margin-bottom: 10px">
Instead of one token at a time...
</p>

<div class="token-parallel">
<div class="token-box active">Token 1</div>
<div class="token-box active">Token 2</div>
<div class="token-box active">Token 3</div>
<div class="token-box active">Token 4</div>
<div class="token-box active">Token 5</div>
<div class="token-box active">Token 6</div>
</div>

<p class="center-text" style="font-size: 0.75em; color: var(--accent)">
All tokens generated and refined simultaneously
</p>

- Start with a <span class="highlight">fixed-length noisy sequence</span> of token embeddings
- Iteratively denoise &mdash; each step refines **all tokens in parallel**
- <span class="highlight2">O(T)</span> denoising steps where T &laquo; n (number of tokens)
- Significant speedup for long outputs &mdash; e.g., generating 512 tokens in ~20 steps instead of 512 forward passes

---

## How Text Diffusion Works Internally

### The Forward & Reverse Process

<div class="two-col">
<div class="col-box">
<h3 style="color: var(--accent)">Forward (Training)</h3>
<ul>
<li>Take a clean text sequence</li>
<li>Gradually add Gaussian noise to token embeddings</li>
<li>At each timestep t, the signal degrades</li>
<li>Model learns: given noise level t, predict the clean tokens</li>
</ul>
</div>
<div class="col-box">
<h3 style="color: var(--accent2)">Reverse (Inference)</h3>
<ul>
<li>Start from pure noise in embedding space</li>
<li>Condition on the input prompt</li>
<li>Denoise step by step</li>
<li>Each step: predict &amp; subtract noise from <strong>all positions</strong></li>
</ul>
</div>
</div>

<p style="font-size: 0.75em; color: #888; text-align: center; margin-top: 15px">
Works in continuous embedding space, not discrete token space
&mdash; tokens are decoded at the final step
</p>

---

## Can Diffusion Models "Think"?

### Chain-of-Thought in Diffusion

- In transformers, "thinking" = generating intermediate reasoning tokens <span class="highlight">sequentially</span>
- Diffusion models refine <span class="highlight2">globally</span> &mdash; the entire output evolves together
- Thinking in diffusion is more like <span class="highlight">iterative refinement</span>:

<div class="flow-diagram" style="margin-top: 15px">
<div class="flow-step">Rough Draft<br /><span class="small">step 1</span></div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Better Structure<br /><span class="small">step 5</span></div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Coherent Logic<br /><span class="small">step 10</span></div>
<div class="flow-arrow">&rarr;</div>
<div class="flow-step">Final Answer<br /><span class="small">step T</span></div>
</div>

- More denoising steps = more "thinking time" &mdash; akin to <span class="highlight">compute-at-inference scaling</span>
- Research area: can the model learn to allocate more refinement steps to harder parts?

---

## Why Transformers Struggle with Image Generation

<div class="two-col">
<div class="col-box">
<h3 style="color: var(--accent)">The Problem</h3>
<ul>
<li>Images have <strong>no natural left-to-right order</strong> &mdash; autoregressive generation is unnatural</li>
<li>A 512x512 image = 262,144 pixels &mdash; generating one at a time is prohibitively slow</li>
<li>Spatial coherence requires <span class="highlight2">global context</span>, not just left-context</li>
<li>Quadratic attention cost on long pixel sequences</li>
</ul>
</div>
<div class="col-box">
<h3 style="color: var(--accent2)">Why Diffusion Fits</h3>
<ul>
<li>Operates on the <strong>entire image at once</strong> &mdash; naturally parallel</li>
<li>Works in <span class="highlight">latent space</span> (e.g., 64x64) not pixel space</li>
<li>Global refinement preserves spatial coherence across the whole image</li>
<li>Fixed compute cost per step regardless of output resolution</li>
</ul>
</div>
</div>

---

## Diffusion vs Transformers for Text

<table class="comparison-table">
<thead>
<tr>
<th></th>
<th>Autoregressive Transformer</th>
<th>Text Diffusion Model</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Generation</strong></td>
<td>Sequential (one token)</td>
<td>Parallel (all tokens)</td>
</tr>
<tr>
<td><strong>Speed scaling</strong></td>
<td>O(n) forward passes</td>
<td>O(T) steps, T &laquo; n</td>
</tr>
<tr>
<td><strong>Editing</strong></td>
<td>Regenerate from edit point</td>
<td>Re-denoise specific regions</td>
</tr>
<tr>
<td><strong>Output length</strong></td>
<td>Variable (natural)</td>
<td>Fixed (must pre-specify)</td>
</tr>
<tr>
<td><strong>Quality</strong></td>
<td>State-of-the-art</td>
<td>Improving rapidly</td>
</tr>
<tr>
<td><strong>Controllability</strong></td>
<td>Via prompting</td>
<td>Via guidance + prompting</td>
</tr>
<tr>
<td><strong>Error handling</strong></td>
<td>Errors cascade forward</td>
<td>Errors corrected in later steps</td>
</tr>
</tbody>
</table>

---

## Diffusion for LLMs: Pros & Cons

<div class="two-col">
<div class="col-box pros">
<h3>Advantages</h3>
<ul>
<li>Parallel generation &mdash; massive speedup for long outputs</li>
<li>Self-correcting: later steps fix earlier errors</li>
<li>Global coherence &mdash; the whole output is refined together</li>
<li>Controllable via classifier-free guidance</li>
<li>Natural fit for editing / infilling tasks</li>
<li>Compute scaling at inference time (more steps = better)</li>
</ul>
</div>
<div class="col-box cons">
<h3>Disadvantages</h3>
<ul>
<li>Fixed output length &mdash; must pre-specify size</li>
<li>Text quality still lags behind top autoregressive models</li>
<li>Harder to do streaming / real-time output</li>
<li>Training is more complex (noise schedules, sampling)</li>
<li>Discrete tokens don't naturally fit continuous diffusion</li>
<li>Less mature ecosystem and tooling</li>
</ul>
</div>
</div>

---

## Notable Text Diffusion Models

- <span class="highlight">Diffusion-LM</span> (Li et al., 2022) &mdash; first controllable text diffusion model
- <span class="highlight">MDLM</span> (Masked Discrete Language Model) &mdash; bridges discrete tokens and diffusion
- <span class="highlight">SEDD</span> (Score Entropy Discrete Diffusion) &mdash; works directly in discrete token space
- <span class="highlight">Mercury</span> (Inception Labs) &mdash; first diffusion LLM with competitive benchmarks, 10x faster generation
- <span class="highlight">Dream / LLaDA</span> &mdash; discrete diffusion that matches autoregressive quality on some benchmarks

<p style="font-size: 0.75em; color: #888; margin-top: 15px">
The field is moving fast &mdash; 2024-2025 saw major breakthroughs
in closing the quality gap.
</p>

---

<!-- .slide: class="section-divider" -->

## Key Takeaways

<div style="text-align: left; max-width: 700px; margin: 30px auto">
<ul style="font-size: 0.85em; line-height: 2">
<li><span class="highlight">AI-on-chip</span> trades flexibility for extreme speed and efficiency &mdash; specialization is the future of inference</li>
<li><span class="highlight">Taalas</span> eliminates the memory wall by unifying compute and storage on a single chip</li>
<li><span class="highlight2">Diffusion models</span> break the autoregressive bottleneck with parallel token generation</li>
<li><span class="highlight2">Thinking in diffusion</span> = iterative global refinement, not sequential token generation</li>
<li>Both represent a shift: from <em>general &rarr; specialized</em> and from <em>sequential &rarr; parallel</em></li>
</ul>
</div>

---

<!-- .slide: class="title-slide" -->

# Thank You

<p style="color: #888; font-size: 0.8em; margin-top: 30px">
Questions?
</p>

<p class="small" style="margin-top: 40px">
References: Taalas &mdash;
<em>The Path to Ubiquitous AI</em> &bull; Li et al. &mdash;
<em>Diffusion-LM</em> (2022) &bull; Inception Labs &mdash;
<em>Mercury</em> (2024)
</p>
