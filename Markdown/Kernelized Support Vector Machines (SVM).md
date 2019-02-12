# Kernelized Support Vector Machines (SVM)

##Feature Mapping

Linear models are great because they are easy to understand and easy to optimize. They suffer because they can only learn very simple decision boundaries. One way of getting a linear model to "behave" non-linearly is to transform the input to higher dimensions, leading to a richer feature space.

This can be demonstrated by a simple example:

![Screen Shot 2019-02-06 at 1.31.36 PM](/Users/ktan/Desktop/Screen Shot 2019-02-06 at 1.31.36 PM.png)

It's clear that **there exists no line that can perfectly separate the data** (in 1D, a boundary is a vertical line given by the equation $x = x_b$, where $x_b$ is the boundary).

Let's try to use feature mapping to solve this. We introduce a function $\phi(x): \mathbb{R} \rightarrow \mathbb{R}^2​$:
$$
\phi(x) = \begin{bmatrix}x \\ x^2\end{bmatrix}
$$
$\phi(x)$ is a function that maps from the first dimension to the second dimension. In this case, the first component of $\phi(x)$ is coordinate for the "$x$-axis" or the absicissa, and the second component of $\phi(x)$ is the coordinate for the "$y$-axis" or the ordinate.

Graphically:

<img src="/Users/ktan/Desktop/Screen Shot 2019-02-06 at 1.58.29 PM.png" width=700px>

By mapping our data to a higher dimension, we find that is is now **linearly separable**, meaning an algorithm like perceptron learning would be able to converge.

How does the hyperplane "translate" down to our original feature space, $\mathbb{R}$? 

- In traditional $xy​$-coordinate systems, the equation of a horizontal line is given by $y = y_0​$, where $y_0​$ is the $y​$-intercept. In our case, the $y​$ variable or ordinate is really $x^2​$. 

- If we allow our hyperplane to intercept the "$x^2​$"-axis at 2 (which is just one of many acceptable hyperplanes), we have the equation $x^2 = 2​$. This equation has two solutions, or $x = \pm \sqrt2​$.

Therefore, graphically, in $\mathbb{R}$ we have:

<img src="/Users/ktan/Desktop/Screen Shot 2019-02-06 at 2.16.37 PM.png" width=500px>

### The Problem with Feature Mapping

The most notable problem when it comes to feature mapping is computational complexity. As you map to higher and higher dimensions, the time (and space) it takes to compute all of the higher-dimensional training examples increases exponentially.

To solve, this we introduce an elegant solution known as Kernels.

## Kernels

The key insight in kernel-based learning is that you can *rewrite* many linear models in a way that doesn't ever require you to explicitly compute the feature map $\phi(x)$. 

- Many machine learning algorithms involve a product of the form $\vec\theta \cdot \phi(x)$ (where in this case, we're using feature mapping). 
- The goal is to rewrite these algorithms such that they only ever depend on dot products between two examples. If we name these examples $\vec{x}, \vec{z}​$, we want to rewrite these algorithms in terms of $\phi(\vec{x}) \cdot \phi(\vec{z})​$.
- **Therefore, the point of kernel-based learning is to try and compute $\phi(\vec{x}) \cdot \phi(\vec{z})$ without ever having to compute $\phi(\vec{x})$ or $\phi(\vec{z})$.**
  - Rather, the only computation we would like to have to do is $\vec{x} \cdot \vec{z}$.

It's easiest to see how this works with an example. Let's revisit the 1-D example we used above. We have the feature map:
$$
\phi(x) = \begin{bmatrix}x \\ x^2\end{bmatrix}
$$
Remembering that $x \in \mathbb{R}$ and $\phi(x) \in \mathbb{R}^2$, **our goal is to try and express $\phi(x) \cdot \phi(z)$ in terms of $x \cdot z$.** Since we are in the first dimension, we simply have $x \cdot z = xz$.

We start by calculating $\phi(x) \cdot \phi(z)$, and try to see if it is possible to simplify it **solely** in terms of $xz$.
$$
\begin{split}
	\phi(x) \cdot \phi(z) &= \begin{bmatrix}x \\ x^2\end{bmatrix} \cdot \begin{bmatrix}z \\ z^2\end{bmatrix} \\
	&= xz + x^2z^2 \\
	&= xz(1 + xz)
\end{split}
$$
And there we have it. Although it seems rather arbitrary, what this example shows is that **it is possible to compute the two-dimensional dot product $\phi(x) \cdot \phi(z)$ by only explicitly computing the one-dimensional dot product $xz$.**

Often, we notate $\phi(\vec{x}) \cdot \phi(\vec{z})$ more simply as $K(\vec{x}, \vec{z})$. We call $K$ the kernel function, and it outputs a scalar.

Let's look at one more example where we start from a kernel function and try to understand the higher-dimensional dot product it represents.

Consider a kernel function for vectors in $\mathbb{R}^2$:
$$
K(\vec{x}, \vec{z}) = (1 + \vec{x} \cdot \vec{z})^2
$$
Given that $\vec{x} = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}$ and $\space \vec{z} = \begin{bmatrix}z_1 \\ z_2\end{bmatrix}$,
$$
\begin{split}
	K(\vec{x}, \vec{z}) &= (1 + x_1z_1 + x_2z_2)^2 \\
	&= (1 + x_1z_1 + x_2z_2)(1 + x_1z_1 + x_2z_2) \\
	&= 1 + x_1z_1 + x_2z_2 + x_1z_1 + x_1^2z_1^2 + x_1x_2z_1z_2 + x_2z_2 + x_1x_2z_1z_2 + x_2^2z_2^2 \\
	&= 1 + 2x_1z_1 + 2x_2z_2 + 2x_1x_2z_1z_2 + x_1^2z_1^2 + x_2^2z_2^2
\end{split}
$$
Again, our goal is to try to express this in terms of $\phi(\vec{x}) \cdot \phi(\vec{z})$. To do this, we look at each summand and we find:
$$
K(\vec{x}, \vec{z}) = \begin{bmatrix}
	1 \\
	\sqrt2x_1 \\
	\sqrt2x_2 \\
	\sqrt2x_1x_2 \\
	x_1^2 \\
	x_2^2
\end{bmatrix} \cdot \begin{bmatrix}
	1 \\
	\sqrt2z_1 \\
	\sqrt2z_2 \\
	\sqrt2z_1z_2 \\
	z_1^2 \\
	z_2^2
\end{bmatrix}
$$
Now, our underlying feature map is obvious:
$$
\phi(\vec{x}) = \begin{bmatrix}
	1 \\
	\sqrt2x_1 \\
	\sqrt2x_2 \\
	\sqrt2x_1x_2 \\
	x_1^2 \\
	x_2^2
\end{bmatrix}
$$
**Again, the crux is that our original function in 2-D, $K(\vec{x}, \vec{z}) = (1 + \vec{x} \cdot \vec{z})^2$, allows us to compute a dot product in a 6-dimensional space without ever computing $\phi(\vec{x})$ and $\phi(\vec{z})$.**

### Polynomial Kernel

The kernel seen in the example above is known as the **polynomial kernel** and has the more general form:
$$
K_d(\vec{x}, \vec{z}) = (1 + \vec{x} \cdot \vec{z})^d
$$
where $d$ is a hyperparameter.

### Gaussian Kernel

The **Gaussian kernel** is interesting in that it actually maps the input vector to a feature space of infinite dimensions. It has the equation:
$$
K(\vec{x}, \vec{z}) = e^{-||\vec{x} - \vec{z}||^2 / 2\sigma^2}
$$
where $\sigma$ is a hyperparameter determining the "width" of the kernel. It's interpreted as the desired standard deviation of the kernel.

### Kernel Criteria

We've already established one version of what makes a valid kernel. That is, $K$ is valid if and only if there exists a function $\phi$ such that $K(\vec{x}, \vec{z}) = \phi(\vec{x}) \cdot \phi(\vec{z})$. 

An alternative criterion is more rooted in mathematics. This property is called **Mercer's condition**, and states that a function $K$ is a valid kernel if $K$ is **positive semi-definite**. This boils down to proving:
$$
\sum_{i=1}^n\sum_{j=1}^nK(\vec{x_i}, \vec{x_j})c_ic_j \geq 0 \space \forall \space c_i, c_j \in \mathbb{R}
$$
This allows us to come up with a number of rules for composing kernels out of other kernels. Let $K_1, K_2%$ be valid kernels on $X​$. Then, the following are valid kernels:

- $K(\vec{x}, \vec{z}) = \alpha K_1(\vec{x}, \vec{z}) + \beta K_2(\vec{x}, \vec{z})$, for $\alpha, \beta \geq 0$.
- $K(\vec{x}, \vec{z}) = K_1(\vec{x}, \vec{z})K_2(\vec{x}, \vec{z})$.
- $K(\vec{x}, \vec{z}) = K_1(f(\vec{x}), f(\vec{z}))$, where $f : X \rightarrow X$.
- $K(\vec{x}, \vec{z}) = g(\vec{x})g(\vec{z})$, where $g : X \rightarrow \mathbb{R}$.
- $K(\vec{x}, \vec{z}) = f(K_1(\vec{x}, \vec{z}))$ where $f$ is a polynomial with positive coefficients.

### TODO: kernelized knn example

## Support Vector Machines

