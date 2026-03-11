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

<!-----

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
</table>-->

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

## See It in Action

<blockquote class="twitter-tweet" data-media-max-width="560"><p lang="en" dir="ltr">the level of innovations in &#39;chip&#39; design lately is wild<br><br>- Taalas, which hardwires AI models directly into silicon, hitting 16,000 tokens/sec by ditching GPUs entirely and baking frozen weights into the transistors themselves (demo below of how fast that is)<br>- Cortical Labs,… <a href="https://t.co/xf0zZp2iKJ">pic.twitter.com/xf0zZp2iKJ</a></p>&mdash; @aaronjmars (@aaronjmars) <a href="https://twitter.com/aaronjmars/status/2029697117133152345?ref_src=twsrc%5Etfw">March 5, 2026</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

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

- More denoising steps = more thinking

<!-----

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
</tbody>
</table>-->

---

## Diffusion for LLMs: Pros & Cons

<div class="two-col">
<div class="col-box pros">
<h3>Advantages</h3>
<ul>
<li>Parallel generation &mdash; massive speedup for long outputs</li>
<li>Self-correcting: later steps fix earlier errors</li>
<li>Global coherence &mdash; the whole output is refined together</li>
</ul>
</div>
<div class="col-box cons">
<h3>Disadvantages</h3>
<ul>
<li>Fixed output length &mdash; must pre-specify size</li>
<li>Text quality still lags behind top autoregressive models</li>
<li>Harder to do streaming / real-time output</li>
<li>Complex training</li>
</ul>
</div>
</div>

---

## See It in Action

<blockquote class="twitter-tweet" data-media-max-width="560"><p lang="en" dir="ltr">The most interesting LLM breakthrough of 2025 isn&#39;t from OpenAI or Anthropic, but from Inception Labs - and it&#39;s completely changing how LLMs work.<br>Inception Labs just released &quot;Mercury&quot; - the first diffusion-based language model that&#39;s generating text at speeds we&#39;ve never seen… <a href="https://t.co/0iLgsSmqBO">pic.twitter.com/0iLgsSmqBO</a></p>&mdash; Nir Gazit (@nir_ga) <a href="https://twitter.com/nir_ga/status/1900178853508624819?ref_src=twsrc%5Etfw">March 13, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

---

<!-- .slide: class="section-divider" -->

## Key Takeaways

- These are two of the ways to get higher tokens/sec
- Makes AI availability cheaper and scales it up

---

<!-- .slide: class="title-slide" -->

# Thank You

<p style="color: #888; font-size: 0.8em; margin-top: 30px">
Questions?
</p>
