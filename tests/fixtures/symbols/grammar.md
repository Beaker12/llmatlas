# Symbol grammar cases

Each line holds one rule; tests/test_list_symbols.py reads the symbols line by line.

Subscript text: $W_{gate}$ and $W_{\text{gate}}$ and $W_\text{gate}$ and $W_{\mathrm{gate}}$.

Named subscripts kept: $d_k$ and $n_h$ and $n_{kv}$ and $n_{\text{train}}$ and $\epsilon_{\text{ls}}$.

Running indices dropped: $x_i$ and $x_{ij}$ and $x_{i-1}$ and $\omega_{2i+1}$ and $q_{m'}$ and $u_t$ and $U_s$ and $\Sigma_{jj}$ and $y_{l}$.

Digit and command subscripts kept: $\beta_1$ and $\lambda_{\max}$.

Wrappers kept: $\mathcal{L}$ and $\mathbf{z}$ and $\mathbb{E}$ and $\mathrm{i}$ and $\mathcal{B}_t$ and $\mathbf{W}_Q$.

Accents kept: $\tilde\rho_i$ and $\tilde{\rho}_i$ and $\bar\rho$ and $\hat{a}$.

Superscripts dropped: $h^{(l)}_i$ and $h_i^{(l)}$ and $n'$ and $C^{a_N}$ and $A^\top$.

Capital superscript labels: $W^{UK}$ and $W^{DKV}$ and $d_h^{R}$ and $V^{UK\top}$.

Transposes and limits: $\sum_{i=1}^{T} x_i^T$ and $\prod_{i=1}^T a_i$ and $\mathbf{1}^T$ and $\left(AB\right)^T$ and $QK^T$.

Other capital superscripts listed: $W_i^Q$ and $\mathbb{R}^{V}$ and $C^{N}$.

Words are names: $PE_{(pos, 2i)}$ and $\sin(pos)$.

Conditions: $x_{<i}$ and $\mathbb{E}_{z \sim p}$ and $w_{1:N}$.

Operators are not symbols: $\log p + \exp(z) + \max_i a_i + \sum_{j=1}^{T} b_j + \operatorname{bucket}(r) + \mathrm{softmax}(S) + \text{loss} + \mathrm{FFN}_e(v) + \text{head}_c + \frac{1}{\tau}\sqrt{g}$.

$$\mathcal{O}(N) + \mathrm{softmax}(QK^\top)$$

```python
x = "$not_math$"
```
