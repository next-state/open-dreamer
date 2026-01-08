## Roadmap
The high level plan is to first implement the entire algorithm, and validate on a toy dataset.
- [x] Causal Tokenizer
- [x] Interactive Dynamics Model
- [x] Imagination Training 

We have finished the validation on a toy dataset, and will now move onto CoinRun.

See implementation progress log [here](https://docs.google.com/presentation/d/1Hc5E-KDZjBwd4HNpFpNoYiuJJTXs15PEDJzw_-VBX3E/edit?usp=sharing).

<details>
<summary><b>Detailed Roadmap</b></summary>
<br>

- [x] Toy Video Dataset generation
- [x] Causal Tokenizer
    - [x] Space-time Axial Attention
    - [x] MAE encoder decoder 
    - [x] MSE loss
    - [x] Training loop
    - [x] LPIPS loss
    - [x] checkpointing
    - [ ] Minor improvements
        - [x] wandb logging
        - [x] cli args / config 
        - [ ] RoPE
        - [x] SwiGLU
        - [x] GQA
        - [x] QK normalization

- [ ] Interactive Dynamics Model
    - [x] Add actions to Toy Video Dataset
    - [x] Add multi-modality support for efficient transformer block
    - [x] setup architecture
        - [x] need to generalize tokenizer preprocessing logic
        - [x] need a "null" action
        - [ ] key / value caching for faster generation
            - [ ] K/V cache across diffusion sampling
            - [ ] K/V cache across timesteps
    - [x] training loop
        - [x] data loading, need to import an encoder and forward the data.
        - [x] shortcut loss function 
        - [x] visualization of predictions
        - [x] optimized loss function 
        - [x] proper checkpointing and loading encoder weights
- [x] Imagination training 
    - [x] Behavior cloning and reward model
        - [x] Update dynamics model architecture to take in agent tokens.
        - [x] Finetune WM on actions and rewards
            - [x] proper handling of terminations, initial rewards, etc.
    - [x] RL Training
        - [x] Initial PMPO update step
        - [x] Visualization helpers for verifying imagination rollouts
        - [x] Speed up imagination generation through JIT
        - [x] Completely fuse all operations into one train step for JIT
        - [ ] Imagination KV caching
- [ ] Clean up code in preparation for CoinRun
    - [x] Remove unused files
    - [ ] Document how to reproduce
- [ ] CoinRun experiment
    - [ ] CoinRun dataloading
        - [ ] MAE training
        - [ ] Dynamics training
        - [ ] Dynamics finetuning
    - [ ] Architecture improvements
        - [ ] Imagination KV caching
        - [ ] Attention logit soft capping
        - [ ] Continue predictor
        - [ ] RoPE embeddings
</details>

## Installation
Use `uv` to create the virtual environment and install dependencies:
```bash
uv sync      # creates .venv and installs packages from uv.lock
source .venv/bin/activate
```
By default, this installs the CPU version of jax. Follow the instructions in the Jax repo to install the GPU version (e.g. `uv pip install "jax[cuda12]"`).

## Dataset generation

The `data.py` module loads recorded trajectory datasets in ArrayRecord format. It outputs a (B,T,H,W,C) batch of data.

We follow the Dreamer convention of indexing for transitions, in which action $a_i$ happens *before* $s_i$. The MDP executes like so:
```
# reset the env. a0, r0, d0 are all "null".
T=0: a0, r0, d0, s0
Agent takes in s0 and outputs a1. Call env.step(a1) to get next r,d,s
T=1: a1, r1, d1, s1
...
T=T-1: a(T-1), r(T-1), d(T-1), s(T-1)
Agent takes in s(T-1) and outputs aT. Call env.step(aT) to get next r,d,s.
T=T: aT, rT, dT, sT
```
With this notation in mind, all initial (T=0) actions, rewards, and dones should be marked invalid and ignored during reward prediction, action prediction, value learning, etc. 


<img src="docs/video.gif" width="500px">
- frames: (B,T,H,W,C), normalized between 0 and 1
- actions: (B,T,|A|), first timestep has dummy action
- rewards: (B, T), reward data, first timestep has dummy reward
- task: (B,), task ID as an integer

 
## Causal Tokenizer
The Causal Tokenizer learns the representation for Dreamer 4.

<img src="docs/step_75900.png" width="300px">

From top to bottom we have: 2 images, their masked variants that are fed to the encoder, then 4 decoder outputs (2 partial decodings, and 2 full decodings). We can see the decoder outputs match the input images well.  

It is a causal transformer encoder, trained through a masked autoencoder loss. Dreamer 4 uses an efficient transformer design which uses axial attention to attend to the space and time dimensions independently. 


<details>
<summary><b>Architecture Details</b></summary>
<br>

Data variables:

| Variable | Description |
|----------|-------------|
| B        | Batch size  |
| T        | Number of timesteps in the video sequence |
| H        | Height of the video frames |
| W        | Width of the video frames |
| C        | Number of channels in the video frames (e.g., 3 for RGB) |
| P        | Patch size for patchifying the video frames |
| N_p      | Number of patches per frame, calculated as (H/P)*(W/P) |
| D        | Dimensionality of each patch, calculated as P*P*C |

Encoder model settings:
| Variable | Description |
|----------|-------------|
| N_l      | Number of latent tokens |
| D_bottleneck | Dimensionality of the bottleneck space |



Data to Encoder steps:
1. Take in video data of shape (B, T, H, W, C). 
2. Patchify with patches of size P, to $(B, T, H/P, W/P, C)$, and then flatten the patches to a sequence $(B, T, N_p, D)$ where $N_p = (H/P)*(W/P)$ and $D = P*P*C$.
3. Prepend learnable latent tokens before each patch token in each timestep to get a batch of sequences, shape $(B, T,  N_l + N_p, D)$, where $N_l$ is the number of latent tokens.
4. Spatial attention: First, let's zoom in on just one timestep with its $N_l$ latent tokens and $N_p$ patch tokens. We do full attention over these tokens. Now, zooming out, this is done on each timestep independently. 
5. Then, we perform causal attention over just the $N_l$ latent tokens to aggregate info across time. In practice, we do causal attention only once every 4 spatial attention layers, to save computation. 
6. Then, we project the last attention block's latent tokens into a lower dimensional bottleneck space, using a linear projection and a tanh. This is the $z$ space in the paper.

Then, we train the $z$ representation using a decoder. The decoder is also a transformer, using the same axial attention idea. 
1. Take in the sequence of $z$ tokens, of shape $(B, T, N_l, D_\text{bottleneck})$. Prepend learnable patch query tokens to each timestep, to get shape $(B, T, N_l + N_p, D_\text{bottleneck})$.
2. Do spatial attention over the $N_l + N_p$ tokens at each timestep independently, then do causal attention over the $N_l$ latent tokens every 4 layers.
3. Finally, project the patch query tokens back to pixel space using a linear layer, and unpatchify to get back to $(B, T, H, W, C)$.
</details>




### Training the Causal Tokenizer
The causal tokenizer is trained using a masked autoencoder loss. We randomly mask out a portion of the input patches with probability $U(0, 0.9)$, and replace the missing portion with a learnable embedding. We then pass the masked input through the encoder and decoder, and train the model with reconstruction loss (MSE) between the original and reconstructed images.

## Interactive Dynamics 
- need to make sure we are handling multiple modalities correctly, where latent tokens can read from everything but other tokens can only attend amongst tokens with the same modality 

Did a fair amount of debugging to improve generations. The way I was doing sampling was incorrect, now the quality is a lot better. We need a variable step size to account for the signal level. The generation quality has been validated with high-fidelity predictions on recorded trajectory datasets.

### Reward / Action prediction
Then, the interactive dynamics transformer is finetuned on action and reward prediction losses. The predicted actions and rewards look fairly accurate over the autoregressive rollout.

### RL in imagination
TD-lambda return computation.

We follow the Dreamer convention of indexing for transitions, where starting at state t, we execute action *t+1* and get reward *t+1* and next state *t+1*. So the transition tuple is indexed as (s[t], a[t+1], r[t+1], s[t+1]) instead of the typical (s[t], a[t], r[t], s[t+1]). Here's the example.
```
# reset the env. a0, r0, d0 are all "null".
T=0: a0, r0, d0, s0
Agent takes in s0 and outputs a1. Call env.step(a1) to get next r,d,s
T=1: a1, r1, d1, s1
...
T=T-1: a(T-1), r(T-1), d(T-1), s(T-1)
Agent takes in s(T-1) and outputs aT. Call env.step(aT) to get next r,d,s.
T=T: aT, rT, dT, sT
```
With this notation in mind, all initial (T=0) actions, rewards, and dones should be marked invalid and ignored during reward prediction, action prediction, value learning, etc. 

Now, our goal is to compute advantages to train the policy. We define the advantage as the TD-lambda return R(s[t], a[t+1]) minus the value V(s[t]). 
Let's say we have an imagination rollout `(s0, a1, s1, a2, s2, ... s[T-1], aT, sT)`, and the corresponding rewards r[t], values v[t], and hidden states h[t]. 

Then, we compute TD-lambda returns from T to 0 like so:
```
for t in range(T-1, -1, -1):
    R[t] = r[t+1] + γ * ((1-λ) * v[t+1] + λ * R[t+1])

which looks like:
R[T] = v[T]
R[T-1] = r[T]   + γ * ((1-λ) * v[T]   + λ * R[T])
...
R[0]   = r[1]   + γ * ((1-λ) * v[1]   + λ * R[1])
```


Then, we can use the TD-lambda returns `R[0...T-1]` to supervise the value prediction head for timesteps 0...T-1, like:
`v[0...T-1] = predict => R[0...T-1]`. Note that we don't include T, since that's just doing `v[T] = predict => v[T]` which is useless.


Then, we call the policy head over all of the hidden states `h[0...T-1]` to get their actions `A` which is an array of length T. But remember these actions will actually correspond to the timesteps 1...T, that is `A[0] = a1`. We skip calling the policy on `h[T]` since we don't have supervision signal for the action `a[T+1]`. 

Then, let's think about how to correspond which TD-lambda return with which action for the PMPO update. `R[0]` is the predicted return from starting at `s0`, executing `a1`, and receiving `r1` as the reward and `s1` as the next state. Then, this means we should correspond `R[0]` with the action `a1` which is actually the first element in `A`. So then it should be straightforward `A[0...T]` gets supervised with `R[0....T]`.
