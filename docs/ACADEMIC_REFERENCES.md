# Aethelnet Core: Theoretical Foundations & Academic References

The Aethelnet architecture synthesizes principles from dynamical systems theory, continuous-time representation learning, epistemic evidence theory, active inference, and cryptographic authenticated data structures.

Below is the formal literature foundation underpinning the Aethelnet Core mathematical engine and its domain dualities.

---

## 1. Continuous-Time Dynamical Systems & Neural ODEs

Aethelnet replaces discrete layer-by-layer feedforward passes with continuous state trajectories governed by Ordinary Differential Equations (ODEs):

$$\frac{dh_i(t)}{dt} = -\tau_i h_i(t) + \tanh\left( \sum_{j \in \mathcal{N}(i)} W_{ij}(t) h_j(t) + I_i(t) \right)$$

* **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).** *Neural Ordinary Differential Equations.* Advances in Neural Information Processing Systems (NeurIPS 2018). [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
  * **Application:** Provides the adjoint sensitivity method and adaptive step-size ODE solvers (`torchdiffeq`, Dormand-Prince `dopri5`) for continuous topological evolution in `aethelnet.liquid_graph`.
* **Chamberlain, B., Rowbottom, J., Gorinova, M. I., Webb, S., Rossi, E., & Bronstein, M. M. (2021).** *GRAND: Graph Neural Diffusion.* International Conference on Machine Learning (ICML 2021). [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
  * **Application:** Establishes the equivalence between deep graph neural networks and continuous diffusion processes on non-Euclidean manifolds, solving oversmoothing and feature collapse.

---

## 2. Liquid Time-Constant (LTC) Networks & Adaptive Decay

To handle irregular, event-based time-series data without fixed sampling rates, Aethelnet incorporates dynamic time-constants ($\tau_i(t)$):

* **Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).** *Liquid Time-constant Networks.* Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
  * **Application:** Enables concept nodes to dynamically adjust their relaxation rate based on incoming sensory perturbation and evidentiary volatility.
* **Lechner, M., Hasani, R., Amini, A., Henzinger, T. A., Rus, D., & Grosu, R. (2020).** *Neural circuit policies enabling auditable autonomous vision.* Nature Machine Intelligence, 2(10), 642–652.
  * **Application:** Sparse, biologically inspired synaptic connectivity patterns that retain causal interpretability.

---

## 3. Active Inference & The Free Energy Principle

The spatial planning and neurosymbolic reasoning engines (demonstrated in ARC-AGI-3 and Sokoban corner-deadlock resolution) derive from active inference:

* **Friston, K. (2010).** *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127–138.
  * **Application:** The agent minimizes variational free energy by alternating between perceptual belief updating and targeted epistemic actions (macro-intent generation).
* **Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).** *Active inference: a process theory.* Neural Computation, 29(1), 1–49.
  * **Application:** Active epistemic foraging: exploratory actions are selected to maximize mutual information and resolve causal ambiguity in unknown environments.

---

## 4. Epistemic Evidence Theory & Multi-Source Fusion

The 4-tier epistemic weighting and contradiction arbitration engines replace naive democratic voting with formal mathematical authority modeling:

* **Dempster, A. P. (1968).** *A generalization of Bayesian inference.* Journal of the Royal Statistical Society: Series B (Methodological), 30(2), 205–232.
* **Shafer, G. (1976).** *A Mathematical Theory of Evidence.* Princeton University Press.
  * **Application:** Underpins the combination of independent evidence sources ($N$ sources) and conflict metric calculation ($\Delta W$) between mutually exclusive propositions.
* **Pearl, J. (1988).** *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference.* Morgan Kaufmann.
  * **Application:** Directional evidentiary propagation and contradiction resolution across causal directed acyclic graphs (DAGs) and multigraphs.

---

## 5. Geometric Deep Learning & Topological Plasticity

* **Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021).** *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
  * **Application:** Formulation of graph manifolds invariant to gauge symmetries, permutation, and spatial translations.
* **Hebb, D. O. (1949).** *The Organization of Behavior: A Neuropsychological Theory.* John Wiley & Sons.
  * **Application:** The continuous weight update equation $\frac{dW_{ij}}{dt} = \eta (h_i h_j) - \gamma W_{ij}$, establishing dynamic synaptic consolidation and pruning.

---

## 6. Authenticated Data Structures & Account Abstraction

Aethelnet establishes trust-minimized historical attestation via cryptographic commitments:

* **Merkle, R. C. (1987).** *A Digital Signature Based on a Conventional Encryption Function.* Advances in Cryptology — CRYPTO ’87, 369–378.
  * **Application:** Deterministic SHA-256 Merkle state root construction across multigraph entities and edges:
    $$\text{StateRoot} = \text{SHA256}\left(\text{app\_id} \;\|\; \bigoplus_{n \in V} \text{Hash}(n) \;\|\; \bigoplus_{e \in E} \text{Hash}(e)\right)$$
* **Buterin, V., et al. (2021).** *EIP-4337: Account Abstraction Using Alt Mempool.* Ethereum Improvement Proposals.
  * **Application:** Gasless, sovereign on-chain state root notarization via `AethelnetPaymaster.sol` and `TheForge.sol`.

---

## 7. Non-Linear Dynamics & Chaos Detection

* **Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).** *Determining Lyapunov exponents from a time series.* Physica D: Nonlinear Phenomena, 16(3), 285–317.
  * **Application:** Measures phase-space divergence and turbulence in high-frequency order flows, dynamically scaling the ODE damping coefficient $\tau_i$.
