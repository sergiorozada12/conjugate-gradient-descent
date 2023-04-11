# Conjugate Gradient Descent Tutorial

This repository provides a Python implementation and walkthrough of gradient descent and conjugate gradient descent algorithms for optimization. The mathematical concepts and code are based on the excellent tutorial found at [Gregory Gundersen's blog](https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/).

## Table of Contents

1. [Introduction](#introduction)
2. [Gradient Descent](#gradient-descent)
3. [Conjugate Gradient Descent](#conjugate-gradient-descent)
4. [Usage](#usage)
5. [References](#references)

## Introduction

Gradient descent and conjugate gradient descent are optimization algorithms used for finding the minimum of a function. Gradient descent is a first-order optimization algorithm that uses the gradient of the function to iteratively move towards the minimum. Conjugate gradient descent is a more advanced method that uses conjugate directions for a more efficient search.

## Gradient Descent

Gradient descent is an iterative optimization algorithm that finds the minimum of a differentiable function by following the negative gradient. In each iteration, the algorithm moves in the direction of the steepest descent, which is given by the negative gradient of the function.

The update rule for gradient descent is:

`x_new = x_old - learning_rate * gradient(x_old)`

where `learning_rate` is a positive scalar that determines the step size.

### Python Implementation

_TODO: Add code snippet for gradient descent implementation_

## Conjugate Gradient Descent

Conjugate gradient descent is an optimization algorithm that improves upon gradient descent by using conjugate directions instead of the steepest descent direction. Conjugate directions are a set of mutually orthogonal directions in the search space that ensure that the algorithm converges in at most N iterations for an N-dimensional quadratic function.

The conjugate gradient descent algorithm involves the following steps:

1. Initialize the starting point `x0` and compute the initial residual `r0 = b - Ax0`.
2. Set the initial search direction `d0 = r0`.
3. For each iteration:
   a. Compute the step size `alpha`: `alpha = (r.T @ r) / (d.T @ A @ d)`
   b. Update the solution: `x_new = x_old + alpha * d`
   c. Update the residual: `r_new = r_old - alpha * (A @ d)`
   d. Compute the new search direction: `d_new = r_new + beta * d_old`, where `beta = (r_new.T @ r_new) / (r_old.T @ r_old)`
   e. Update the old values: `x_old = x_new`, `r_old = r_new`, and `d_old = d_new`

### Python Implementation

````
def cgd(x0, A, b):
    x = x0
    r    = b - A @ x
    d    = r
    rr   = r.T @ r
    xs   = [x]
    for _ in range(A.shape[0]):
        Ad     = A @ d
        alpha  = rr / (d.T @ Ad)
        x      = x + alpha * d
        r      = r - alpha * Ad
        rr_new = r.T @ r
        beta   = rr_new / rr
        d      = r + beta * d
        rr     = rr_new
        xs.append(x)
    return np.array(xs)
````

## Usage

_TODO: Add usage instructions, including any necessary installation steps, input/output formats, and example code_

## References

- [Conjugate Gradient Descent Tutorial by Gregory Gundersen](https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/)
- [Conjugate Gradient Descent on Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method)