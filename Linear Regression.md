# Linear Regression

## Cost Function

$$
\begin{split}
	J(\vec\theta) &= \frac{1}{2m}\sum_i(\vec{x_i} \cdot \vec\theta - y_i)^2 \\
	&= \frac{1}{2m}\sum_i((x_{i,1}\theta_1 + x_{i,2}\theta_2 + \dots + x_{i,n}\theta_n) - y_i)^2
\end{split}
$$

To take the gradient of $J(\vec\theta)$, we consider the partial derivative with respect to $\theta_k$ for some $k \in 1 ... n$.
$$
\frac{\partial J}{\partial \theta_k} = \frac{1}{m}\sum_i(\vec{x_i} \cdot \vec\theta - y_i)x_{i,k}
$$
Since this holds for every $k \in 1 ... n$, we have:
$$
\nabla J(\vec\theta) = \frac{1}{m}\sum_i(\vec{x_i} \cdot \vec\theta - y_i)\vec{x_i}
$$
Note that $\nabla J(\vec\theta)​$ may also appear as just $J'(\vec\theta)​$ or $\frac{\partial J}{\partial \vec\theta}​$.

---

### Matrix Version

To derive the matrix version, we require some intermediary lemmas:

**<u>Proposition 1</u>**: If $y(\vec\beta) = \vec{x} \cdot \vec\beta$, and $\vec{x}$ does not depend on $\vec\beta$, then:
$$
\frac{\partial y}{\partial \vec\beta} = \vec{x}
$$
(Note that $y(\vec\beta)$ is a scalar, but $\frac{\partial y}{\partial \vec\beta}$, its derivative/gradient, is a vector!) 

**<u>Proof</u>**:
$$
y(\vec\beta) = \sum_i x_i\beta_i = x_1\beta_1 + x_2\beta_2 + ... + x_i\beta_i \\
\text{We take the partial derivative with respect to }\beta_j\text{ for some }j \in 1 ... i: \\
\frac{\partial y}{\partial \beta_j} = x_j. \\
\text{Since the }j^{\text{th}}\text{ component of }\frac{\partial y}{\partial \vec\beta}\text{ is }x_j\text{ for all }j, \\
\therefore \frac{\partial y}{\partial \vec\beta} = \vec{x}
$$

---

**<u>Proposition 2</u>**: Let the scalar $\alpha$ be defined by $\alpha = \vec{y}^T\bold{A}\vec{x}$ where:

- $\vec{y}​$ is $m \times 1​$
- $\vec{x}​$ is $n \times 1​$
- $\bold{A}​$ is $m \times n​$, and independent of $\vec{x}​$ and $\vec{y}​$

Then:
$$
\frac{\partial \alpha}{\partial \vec{x}} = \vec{y}^T\bold{A} \space \text{ and } \space \frac{\partial \alpha}{\partial \vec{y}} = \vec{x}^T\bold{A}^T.
$$
**<u>Proof</u>**:

Define $\vec{w}^T = \vec{y}^T\bold{A}$ (so $\vec{w}$ is an $n \times 1$ column vector).

Then:
$$
\alpha = \vec{w}^T\vec{x}
$$
It follows from Proposition 1 that:
$$
\therefore\frac{\partial \alpha}{\partial \vec{x}} = \vec{w}^T = \vec{y}^T\bold{A}.
$$
To prove the next part, we note that since $\alpha$ is a scalar:
$$
\alpha = \alpha^T = \vec{x}^T\bold{A}^T\vec{y}, \\
\therefore\frac{\partial \alpha}{\partial \vec{y}} = \vec{x}^T\bold{A}^T.
$$

---

**<u>Proposition 3</u>**: Let the scalar $\alpha$ be defined by $\alpha = \vec{x}^T\bold{A}\vec{x}$ where:

- $\vec{x}$ is $n \times 1$
- $\bold{A}$ is $m \times n$ and independent of $\vec{x}​$

Then:
$$
\frac{\partial \alpha}{\partial \vec{x}} = \vec{x}^T(\bold{A} + \bold{A}^T)
$$
**<u>Proof</u>**:

By definition, we have:
$$
\alpha = \sum_{j=1}^n\sum_{i=1}^nA_{i,j}x_ix_j, \text{ so:} \\
\frac{\partial \alpha}{\partial x_k} = \sum_{j=1}^nA_{k,j}x_j + \sum_{i=1}^nA_{i,k}x_i \text{ for all }k \\
\therefore \frac{\partial \alpha}{\partial \vec{x}} = \vec{x}^T\bold{A}^T + \vec{x}^T\bold{A} = \vec{x}^T(\bold{A}^T + \bold{A}).
$$
**<u>Notes</u>**:

- When taking the derivative with respect to the ${x_k}^{\text{th}}$ component, $\frac{\partial}{\partial x_k}(A_{i,j}x_ix_j)$ is non-zero if and only if $i = k$ or $j = k$.

- To understand $\sum_{j=1}^nA_{k,j}x_j​$, think that for a fixed $k​$ we are "iterating" through the $k^{\text{th}}​$ **row** of the matrix $\bold{A}​$ and multiplying the $j^{\text{th}}​$ element of that row with the $j^{\text{th}}​$ element of $\vec{x}​$, which is the same as $\vec{x}^T\bold{A}^T​$.

- To understand  $\sum_{i=1}^nA_{i,k}x_i$, think that for a fixed $k$ we are "iterating" through the $k^{\text{th}}$ **column** of the matrix $\bold{A}$... so we have the same as $\vec{x}^T\bold{A}$.

- If $\bold{A}$ is symmetric,
  $$
  \frac{\partial \alpha}{\partial \vec{x}} = \vec{x}^T(\bold{A} + \bold{A}) = 2\vec{x}^T\bold{A}.
  $$

---

#### Back to Linear Regression...

$$
J(\vec\theta) = \frac{1}{2n}||(\bold{X}\vec\theta - \vec{y})||
$$

If there are $m$ training samples and $n$ features, then:

- $\bold{X}$ is $m \times n$. A **row** represents 1 training sample
- $\vec\theta$ is $n \times 1$
- $\vec{y}​$ is $m \times 1​$

$$
\begin{split}
	J(\vec\theta) &= \frac{1}{2n}(\bold{X}\vec\theta - \vec{y})^T(\bold{X}\vec\theta - \vec{y}) \\
	&= \frac{1}{2n}(\vec\theta^T\bold{X}^T - \vec{y}^T)(\bold{X}\vec\theta - \vec{y}) \\
	&= \frac{1}{2n}(\vec\theta^T\bold{X}^T\bold{X}\vec\theta - \vec\theta^T\bold{X}^T\vec{y} - \vec{y}^T\bold{X}\vec\theta + \vec{y}^T\vec{y})
\end{split}
$$

We wish to calculate $\nabla J(\vec\theta)​$.
$$
\begin{split}
	\nabla J(\vec\theta) &= \frac{1}{2n}(2\vec\theta^T\bold{X}^T\bold{X} - \vec{y}^T\bold{X} - \vec{y}^T\bold{X} + 0) \\
	&= \frac{1}{n}(\vec\theta^T\bold{X}^T\bold{X} - \vec{y}^T\bold{X})
\end{split}
$$
where we have used the fact that $\bold{X}^T\bold{X}$ is a symmetric matrix, and is independent of $\vec\theta$ and $\vec{y}$. 

###Closed-Form Solution

We set $\nabla J(\vec\theta)$ to $0$ to calculate the closed-form solution:
$$
\begin{split}
	0 &= \frac{1}{n}(\vec\theta^T\bold{X}^T\bold{X} - \vec{y}^T\bold{X}) \\
	\vec\theta^T\bold{X}^T\bold{X} - \vec{y}^T\bold{X} &= 0 \\
	\vec\theta^T\bold{X}^T\bold{X} &= \vec{y}^T\bold{X} \\
	\vec\theta^T &= \vec{y}^T\bold{X}(\bold{X}^T\bold{X})^{-1} \\
\end{split} \\~\\
\therefore\vec\theta = (\bold{X}^T\bold{X})^{-1}\bold{X}^T\vec{y}
$$
where we have used the fact that $\bold{X}^T\bold{X}​$ is invertible, and that the inverse of a symmetric matrix is also symmetric.



