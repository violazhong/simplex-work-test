## Question 2
Make a pre-registered prediction (honor code) as to what geometry the activations should take, 
and how it should change with context position and across layers. Derive this prediction mathematically as far as you can, 
and separately give your intuition for what the geometry should look like. You will not be penalized for getting this wrong—we are interested in both your formal reasoning and your geometric intuition. Are there multiple possible geometries you can think of?

## My answer
I realize that 3 mutually exclusive 3-state HMMs are mathematically equivalent to a single 9-state HMM (the 9 states are mutually exclusive as well), and the transition matrix would have a decomposable block-diagonal structure. Then the key question is whether transformers can discover and use this decomposition. So here, I present two competing hypotheses.

### Hypothesis 1: Direct-Sum Representation (Dedicated Subspaces)
#### Mathematical reasoning 
The model allocates K orthogonal 2D subspaces, one per component. Each subspace contains the Mess3 belief geometry for that component, scaled by the posterior weight. This is lossless: the full mixture prediction can be recovered from the K scaled belief vectors at all context positions. Total effective dimensionality: 2K = 6. This is the natural analog of the paper's Factored World Hypothesis, adapted from tensor-product to direct-sum structure.
#### Geometric intuition
Three orthogonal planes in activation space, each containing one component's fractal belief triangle. A sequence's activation projects onto all three planes simultaneously, with the magnitude of the projection encoding the posterior. In PCA, the top 6 PCs form three component-specific pairs.
#### Context position dynamics
Effective dimensionality stays at ~6 across positions, since the model maintains all three belief states throughout. What changes is the distribution of norm across subspaces: diffuse early, concentrated in one subspace late (posterior has converged). Across layers, earlier layers build up the differential scaling (component discrimination) while later layers refine within-component belief geometry.

### Hypothesis 2: Multiplexer Representation (Dimensional Reuse)
#### Mathematical reasoning
The model exploits mutual exclusivity aggressively: since only one component is ever active, it never needs to track more than one belief state at a time. It factors the representation into a (K−1)-dimensional routing signal encoding and a single shared 2D belief canvas for the conditional state. Total effective dimensionality: (K−1) + 2 = 4. But this is lossy when the posterior is diffuse — the shared canvas cannot simultaneously represent K distinct belief states, so the full mixture prediction is irrecoverable at early context positions. It becomes effectively lossless once the posterior concentrates on one component.
#### Geometric intuition
A 2D "meta-simplex" whose vertices represent certainty about each component, orthogonal to a single shared 2D belief canvas. The meta-simplex acts as a gate: its coordinates dictate which transition dynamics are applied on the shared canvas. All three components' belief geometries overlap in the same 2D subspace, distinguished only by the routing dimensions.
#### Context position dynamics
Early positions: the routing signal is uncertain, the shared canvas holds an imprecise mixture — this is where lossiness bites. Late positions: the routing signal is sharp, and the canvas tracks the identified component's belief state accurately. Effective dimensionality is ~4 throughout, but prediction quality improves with context as routing sharpens. Across layers, attention heads in early layers likely handle component discrimination (populating the meta-simplex), while later layers apply component-conditional belief updates on the shared canvas.
