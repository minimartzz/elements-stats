# 2. Overview of Supervised Learning

Goal of supervised learning is to use the inputs to predict the values of the outputs

- predictors = inputs = independent variables = features
- responses = outputs = dependent variables

# Variable Types & Terminology

- **Quantitative variables** (or continuous variables) — features in this set are just numbers
- **Qualitative variables** (categorical or discrete variables) — features in this set have no fixed order and are distinct, descriptive labels rather than numbers
    - Usually represented numerically by codes
- **Ordinal variables** — features are also descriptive labels, but *order* matters

# Simple Approaches to Prediction

2 simple but powerful prediction methods:

1. *Linear model fit by least squares*
2. *k-nearest-neighbours*

## Linear Models and Least Squares

Given a vector of inputs: $X^T = (X_1,X_2, . . . ,X_p)$, we predict the output $Y$ via the model

$$
\hat{Y}=\hat{\beta}_0+\sum_{j=1}^pX_j\hat{\beta}_j
$$

**Method of Least Squares**: Pick the coefficients $\beta$ that minimize the residual sum of squares

![Reference: [https://bi3mer.github.io/blog/post_30/biemer_least_squares.pdf](https://bi3mer.github.io/blog/post_30/biemer_least_squares.pdf)](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled.png)

Reference: [https://bi3mer.github.io/blog/post_30/biemer_least_squares.pdf](https://bi3mer.github.io/blog/post_30/biemer_least_squares.pdf)

Consider 2 scenarios:

1. **Scenario 1:** The training data in each class were generated from bivariate Gaussian distributions with uncorrelated components and different means
2. **Scenario 2:** The training data in each class came from a mixture of 10 low variance Gaussian distributions, with individual means themselves distributed as Gaussian.

Least squares optimizing is the optimal solution to scenario 1 but not scenario 2. Scenario 2 requires a nonlinear solution

## Nearest-Neighbours Method

Use observations in training set $\tau$ closes in input space to $x$ to form $\hat{Y}$. k-nearest neighbours fit:

$$
\hat{Y}(x)=\frac{1}{k}\sum_{x_i\in N_k(x)} y_i
$$

Find the k observations with $x_i$ closest to $x$ in input space, and average their responses. In classification, set the value to a specific class if $\hat{Y} > 0.5$

Error on the training data should be approximately an *increasing function* of $k$ and will always be 0 for $k=1$

### A note on the number of parameters for both

The number of parameters used in nearest-neighbours is 1 i.e $k$, while for least square is the number of coefficients i.e $p$. Which indicates that there are more parameters to fit for least squares.

However, the *effective* number of parameters for nearest-neighbours is $N/k$ since if all neighbourhoods are not overlapping, there would be 1 mean to fit for each one.

## From Least Squares to Nearest Neighbours

Comparing both methods:

- Least Squares — smooth, stable to fit (low variance, high bias)
- Nearest-neighbours — adaptable to situation (high variance, low bias)

A large subset of popular techniques are variants of these 2 methods. Below are some techniques used to enhance and augment these methods:

1. **Kernel methods** use weights that decrease smoothly to zero with distance from the target point, rather than effective 0/1 weights used by k-nearest neighbour
2. In high-dimensional spaces the distance **distance kernels** are modified to emphasize some variables more than others
3. **Local regression** fits models by locally weighted least squares, rather than fitting constants locally
4. Linear models fit to a **basis expansion** of the original inputs allow arbitrarily complex models.
5. **Projection pursuit** and **neural network** models consist of sums of nonlinearly transformed linear models

# Statistical Decision Theory

Most model frameworks work similarly. There is a real valued random input vector $X\in\mathbb{R}^p$ and an output variable $Y\in\mathbb{R}$, with a joint distribution of $Pr(X, Y)$. Find a function $f(x)$ for predicting $Y$, formulated by a loss function $L(Y, f(x))$

Using the squared error loss: $(Y-f(x))^2$

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%201.png)

The conditional expectation, also known as the regression function. Thus the best prediction of $Y$ at any point $X = x$ is the conditional mean, when best is measured by average squared error.

## For Nearest Neighbours

Approximates the above solution by choosing the set of $x_i$ that is closest to $x$ in the formula:

$$
\hat{f}(x)=Avg(y_i|x_i\in N_k(x))
$$

2 approximations happen here:

- Expectation is approximated by averaging over sample data
- Conditioning at a point is relaxed to condition on some region “close” to the target

For large training sample sizes, the points in the neighbourhood are likely to be close to $x$, and $k$ gets large, the average will get more stable.

Another observation is that as $N,k \rightarrow \infty$, $\hat{f}$ is the conditional expectation of Y given X.

*Rate of convergence decreases as the dimension increases*

## For Linear Regression

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%202.png)

Solution is not conditioned on $X$. Use the functional relationship described by the linear sum to pool over values of X, averaging over the training data

### Key differences between the methods

Both methods approximate conditional expectations using averages, but here are the differences in approach:

- Least squares assumes $f(x)$ is well approximated by a globally linear function
- k-nearest neighbours assumes $f(x)$ is well approximated by locally constant functions

Many modern techniques are model-based i.e similar to least squares, but are more flexible.

Other loss functions also include $L_1: E|Y-f(x)|$ which evaluates to: $\hat{f}(x)=median(Y|X=x)$

## For Classification Problems

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%203.png)

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%204.png)

This solution is the *Bayes Classifier,* which classifies to the most probable class, using the conditional (discrete) distribution **$P(G|X)$

➡️ The error rate of the Bayes classifier is called the *Bayes rate. The Bayes rate is the lowest possible error rate that can be achieved by any classifier for a given problem.* 

<aside>
💡 **Bayes Rate**

---

The error rate of the Bayes classifier. The Bayes classifier is the optimal classifier under the assumption that true distribution of features are known.

- Bayes rate provides a benchmark for evaluating performance of real-world classifiers — error rates that are close are considered good
- It represents the lowest possible error rate
</aside>

# Local Methods in High Dimensions

**Curse of dimensionality** — Approach breaks down in high dimensions

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%205.png)

**Requires knowing a large proportion of observations** to accurately predict

Given a $p$-dimensional unit hypercube, with each side being uniformly distributed. To capture a fraction $r$ of the total volume of the cube, what would be the expected value of each edge:

$$
e_p(r)=r^{(1/p)}
$$

Since each side multiplied together would give the total volume of $r^p$

if $p=10$ i.e 10 dimensions. $e_{10}(0.01)=0.63$ and $e_{10}(0.1)=0.80$ which implies to capture 1% or 10% of the total volume (or total observations) it would require knowing 63% and 80% of the range of each input variable.

⚠️ Reducing $r$ does not help reduce the amount of observations required

**All sample points are close to an edge of the sample** meaning hard to determine a clear decision boundary

The median distance from the origin of a unit sphere with $p$-dimensions and $N$ data points is:

$$
d(p,N)=(1-\frac{1}{2}^{1/n})^{1/p}
$$

if $N=500$ and $p=10$ the distance will be $0.52$ which is more than halfway to the boundary. This indicates that most points are closer to the boundary than to any other point

**Sampling density is proportional to $N^{1/p}$** means to obtain a dense sample, A LOT more data is required for each point.

The sample size required for the same sampling density is much much more. In high dimensions all feasible training samples sparsely populate the input space.

💡 The complexity of functions of many variables can grow exponentially with the data

*Difficult to understand pages 40-46

# Statistical Models and Function Approximations

Find a useful approximation of $\hat{f}(x)$ to the function $f(x)$ that underlies the predictive relationship between inputs and outputs

- 2 important considerations that affect the accuracy of the approximation
    1. **Dimensionality of input space** — data points are more spread out, resulting in large errors
    2. **Special structure is known to exist** — able to reduce both bias and variance of estimates significantly

## Statistical Model for Joint Distribution $Pr(X, Y)$

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%206.png)

The additive nature of the model is a useful approximation of the truth and $\varepsilon$ represents some unmeasured errors that can contribute to $Y$. In reality the relationship of the input-output pairs are not so deterministic, but the use of $\varepsilon$ helps to approximate these errors.

The assumption is that errors are $i.i.d$ is not necessary by making adjustments to the model.

E.g set $Var(Y|X=x) = \sigma(x)$ allows for heteroscedasticity since both mean and variance are conditioned on $X$, therefore independence is no longer a requirement

Additive models are usually not used for qualitative outputs, but applying conditional variance works in a similar manner.

## Supervised Learning

Supervised learning attempts to learn $f$ by modifying it’s input → output relationship $\hat{f}$ in response ot differences in $y_i - \hat{f}(x_i)$ between the original and generated outputs. Process is called *learning by example*

- Training observations: $\Tau = (x_i, y_i), i=1, ..., N$
- Observations: $x_i$
- Model outputs: $\hat{f}(x_i)$

## Function Approximation

Data pairs $\{x_i, y_i\}$ are viewed as points in the dimensional space, while the function $f(x)$ captures the relationship of the data through a function

> **Goal**: Obtain a useful approximation to $f(x)$ for all $x$ in some region of ${\rm I\!R}^p$ given the representation in $\Tau$
> 

We treat supervised learning as a problem in function approximation — encourages use of geometrical and probabilistic concepts. 

Many approximations involve an associated set of parameters $\theta$ that is modified to fit the data.

<aside>
➡️ **Example of an approximation — Linear Basis Expansion**

---

$h_k$ is a set of functions or transformations of input vector $x$. Find the set of $\theta$ that the fitted surface gets as close to the observed points as possible

$$
f_\theta(x)=\sum\limits_{k=1}^Kh_k(x)\theta_k
$$

By minimizing the $RSS$ we get a closed for solution if the basis function does not have hidden parameters. Otherwise a numerical or iterative method is required.

![Objective of function approximation is to get the hyperplane that minimizes the distance to observed points](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%207.png)

Objective of function approximation is to get the hyperplane that minimizes the distance to observed points

</aside>

### Maximum likelihood estimation

A more general approach to error estimation (besides least squares) is **maximum likelihood estimation.** Assumes that the most reasonable value for $\theta$ are those for which the probability of the observed sample is the largest

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%208.png)

[Derivative of log-likelihood function for Gaussian distribution with parameterized variance](https://mathoverflow.net/questions/449798/derivative-of-log-likelihood-function-for-gaussian-distribution-with-parameteriz)

# Structured Regression Models

This section introduces more structured approaches that can make more efficient use of the data, compared to local methods that estimate functions at a point (e.g nearest-neighbours).

## Difficulty of the problem

Why do we need such classes?

For a given function

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%209.png)

If the sample size $N$ was sufficiently large, the solution would all tend to the limiting conditional expectation. But to obtain useful results for finite $N$, restrictions must be placed upon the eligible solution, to considerations outside that of data.

- Restrictions can be imposed in many ways like parametric representation of $f_\theta$ or built into the learning method.
- Restrictions imposed that lead to a unique solution do not remove the ambiguity of the wide variety of solutions i.e there are infinitely many possible restrictions — **it is a choice of the constraint**
- Constraints are described as *complexity restrictions* usually imposing some kind of behaviour in small neighbourhoods of the input space

> For all input points $x$ sufficiently close to each other in some metric, $\hat{f}$ exhibits some special structure such as near constant, linear or low-order polynomial behaviour. The estimator is then obtained by averaging or polynomial fitting in that neighbourhood
> 
- Larger the size of the neighbourhood, stronger the constraint
- Nature of constraint depends on the metric used
    - Some methods (kernel, local regression, tree-based) directly specify the metric and size of neighbourhood
    - Other methods (splines, neural networks and basis-function) implicitly define neighbourhoods of local behaviour
- Any method that attempts to produce locally varying functions in isotropic neighbourhoods will run into the dimensionality problem

# Classes of Restricted Estimators

Nonparametric regression techniques fall into classes depending on the nature of the restriction imposed. Each class has one or more parameters (smoothing parameters) that **control the effective size of the local neighbourhood**

## Roughness penalty and bayesian methods

Penalizing $RSS(f)$ with a roughness penalty (complexity penalty)

🗯️ Remember L1, L2 regularization methods

$$
PRSS(f;\lambda)=RSS(f)+\lambda J(f)
$$

User-selected functional $J(f)$ will be large for functions $f$ that vary too rapidly over small regions of the input space i.e penalize overfitting

Example is the cubic smoothing spline - penalize least-square creiterion

$$
PRSS(f;\lambda)=\sum_{i=1}^N(y_i-f(x_i))^2+\lambda\int[f''(x)]^2dx
$$

- Penalty functions $J$ can be constructed for functions in any dimension
- Penalty functions (or regularization methods) express the belief that the type of functions exhibit a type of smooth behaviour

## Kernel methods and local regression

**Specify the nature of the local neighbourhood** and class of regular functions fitted locally. Local neighbourhood is specified by a *kernel function* $K_\lambda(x_0, x)$ which assigns weights to point $x$ in the region around $x_0$ ($x_0$ is the output of the kernel)

![Untitled](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%2010.png)

## Basis functions and dictionary methods

Linear and polynomial expansions. $h_m$ is a function of the input $x$. Linear refers to the action of the parameters $\theta$

$$
f_\theta(x)=\sum_{m=1}^M\theta_mh_m(x)
$$

Basis functions are a set of functions that combine to form piecewise polynomials that try and predictions. 

- $K$ polynomial splines; $M$ spline basis functions; determined by ($M-K$ knots)
    - *knots* are points at which the individual functions are connected

Basis functions are also known as *dictionary methods*, availability of a possibly infinite set of dictionary of candidate functions to choose from

# Model Selection and the Bias-Variance Tradeoff

As the model complexity increases, the variance tends to increase and the squared bias decreases. Choose the model complexity to trade bias off with variance to **minimize the test error**. 

![Example: Using training error to estimate the test error as model complexity is  varied](2%20Overview%20of%20Supervised%20Learning%20ddf0c61d9ac54ec69f822f23a6ecd957/Untitled%2011.png)

Example: Using training error to estimate the test error as model complexity is  varied

Too much fitting, the model adapts too closely to the training data and does not generalize well i.e $\hat{f}(x_0)$ will have large variance to reflect the complexity of the function, vice versa