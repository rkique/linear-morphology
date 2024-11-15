### Linear Relational Decoding of Morphological Relations in Language Models 

This is the repository accompanying the paper "Jacobian Relational Approximation Captures Morphology". ChatGPT was referenced for data visualization and processing purposes. It was also used to demonstrate features from libraries from which experiment code was written.

### Directory Layout

- *baukit*: A frozen copy of https://github.com/davidbau/baukit, used to trace model hidden states and calculate the Jacobian.

- *data*: Where Jacobians + biases are written to.
    - `Make_Weights_GPT-J.py`, `Make_Weights_Llama.py`: scripts for making Jacobians
    - `TestLREJacobian.py`,`TestLREJacobian_Llama.py`: scripts for evaluating different approximators.

- *experiments*: Prototypes for the above and other experiments, including linear projections in `IdentifyingBiasInVecSpace.ipynb`.

- *llra*: Three helper modules written for this project.
    - `build.py`: Contains essential functions:
        - `get_object` (decodes object hidden state to get token predictions)
        - `get_hidden_state` (gets the hidden state for a prompt and subject at any layer)
        - `get_hidden_states`.
        - An unused "LLRA" (Layerwise Linear Relational Approximator) class
    - `jacobian.py`: Contains unused `get_jacobians`.
    - `viz.py`: Contains essential functions for linear projection, including `proj_matrices`, `make_basis`, and `viz`.

- *lre*: The original LRE code from Hernandez et. al. "Linearity of Relation Decoding in Transformer LMs" (2023). The original code creates weights & biases and tests them in the same process. This version saves weights & biases to a folder, which can be tested separately.

- *reference*: Old scripts imported from various sources for understanding purposes.

- *results*: The final results for GPT-J and Llama, as well as some older results.

