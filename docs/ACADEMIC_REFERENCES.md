# Academic References & Theoretical Foundations

The Aethelnet Observer's visualization engine and the underlying LGNN (Liquid Graph Neural Network) architecture draw heavily from recent advancements in continuous-time representation learning, dynamical systems, and non-Euclidean graph topology.

Below are the core academic references that form the mathematical and conceptual backbone of this project.

## 1. Continuous-Time Neural Networks (Neural ODEs)
The shift from discrete-step execution (traditional neural networks) to continuous-time evolution in Aethelnet Core is based on the seminal work in Neural Ordinary Differential Equations.
* **Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).** *Neural Ordinary Differential Equations.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
  * **Application:** Forms the basis for the `evolve_topology` ODE solver inside `aethelnet-core`, allowing the node activations to flow continuously.

## 2. Liquid Time-Constant Networks
To allow individual nodes to adapt their decay rates ($\\tau_i$) dynamically based on external stimuli (like the `ButterflySensor` detecting market chaos), we utilize principles from Liquid Time-Constant (LTC) networks.
* **Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).** *Liquid Time-constant Networks.* Proceedings of the AAAI Conference on Artificial Intelligence. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
  * **Application:** Enables the Aethelnet Swarm to process irregular, event-based time-series data (e.g., asynchronous tick data and OSINT signals) without fixed sampling rates.

## 3. Geometric Deep Learning & Graph Topology
The 3D WebGL Galaxy visualization (CodeSpider Blueprint) maps the high-dimensional latent space into a 3D coordinate system. The dynamic pruning and bridging of connections use principles of Geometric Deep Learning.
* **Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021).** *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* [arXiv:2104.13478](https://arxiv.org/abs/2104.13478)
  * **Application:** The non-Euclidean state space where nodes drift and form Hebbian bridges.

## 4. Decentralized Byzantine Fault Tolerance
The `aethelnet-contracts` Forge governance and the P2P swarm mesh require asynchronous validation under chaotic conditions.
* **Castro, M., & Liskov, B. (1999).** *Practical Byzantine Fault Tolerance.* Proceedings of the Third Symposium on Operating Systems Design and Implementation (OSDI).
  * **Application:** The underlying consensus assumptions for the Aethelnet P2P Mesh when transmitting `OSINT_CONVICTION` and `INTELLIGENCE` payloads.

## 5. Chaotic Systems (Butterfly Sensors)
The `ButterflySensor` in the observer's `satellite_node.py` calculates Lyapunov exponents to detect sensitive dependence on initial conditions.
* **Wolf, A., Swift, J. B., Swinney, H. L., & Vastano, J. A. (1985).** *Determining Lyapunov exponents from a time series.* Physica D: Nonlinear Phenomena.
  * **Application:** Measuring the chaos/divergence rate of real-time market data to dynamically trigger node activation within the LGNN.
