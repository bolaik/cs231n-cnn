---
output:
  pdf_document: default
  html_document: default
---
## Batch Normalization: alternative backward
In class we talked about two different implementations for the sigmoid backward pass. One strategy is to write out a computation graph composed of simple operations and backprop through all intermediate values. Another strategy is to work out the derivatives on paper. For the sigmoid function, it turns out that you can derive a very simple formula for the backward pass by simplifying gradients on paper.

Surprisingly, it turns out that you can also derive a simple expression for the batch normalization backward pass if you work out derivatives on paper and simplify. After doing so, implement the simplified batch normalization backward pass in the function `batchnorm_backward_alt` and compare the two implementations by running the following. Your two implementations should compute nearly identical results, but the alternative implementation should be a bit faster.

## Draft for the solution
So this, time we want to find $\frac{dL}{d\gamma}$, $\frac{dL}{d\beta}$ and $\frac{dL}{dx}$ with
$$ y = \gamma \hat{x}-\beta$$ where
$$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} = (x-\mu)(\sigma^2+\epsilon)^{-1/2}$$

Therefore, we note for the following that

$$y_{kl} = \gamma_l \hat{x_{kl}}-\beta_l$$
and
$$\hat{x_{kl}}=(x_{kl}-\mu_l)(\sigma_l^2+\epsilon)^{-1/2}$$
where
$$\mu_l = \frac{1}{N}\sum_p x_{pl}$$
and 
$$\sigma_l^2 = \frac{1}{N}\sum_p \left(x_{pl}-\mu_l\right)^2$$

Let's begin by the easy one !

\begin{eqnarray}
\frac{dL}{d\gamma_j} &=& \sum_{kl}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\gamma_j}\\
&=& \sum_{kl}\frac{dL}{dy_{kl}}x_{kl}\delta_{lj}\\
&=& \sum_{k}\frac{dL}{dy_{kj}}x_{kj}
\end{eqnarray}

For $\beta$ we have
\begin{eqnarray}
\frac{dL}{d\beta_j} &=& \sum_{kl}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\beta_j}\\
&=& \sum_{kl}\frac{dL}{dy_{kl}}\delta_{lj}\\
&=& \sum_{k}\frac{dL}{dy_{kj}}
\end{eqnarray}

Ok. Let's start the serious one.
\begin{eqnarray}
\frac{dL}{dx_{ij}} &=& \sum_{kl}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{dx_{ij}}\\
&=& \sum_{kl}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\hat{x}_{kl}}\frac{d\hat{x}_{kl}}{dx_{ij}}
\end{eqnarray}
where $$\hat{x_{kl}}=(x_{kl}-\mu_l)(\sigma_l^2+\epsilon)^{-1/2}$$.
First, we have:
$$ \frac{dy_{kl}}{d\hat{x}_{kl}} = \gamma_l$$
and
\begin{eqnarray}
\frac{d\hat{x}_{kl}}{dx_{ij}} = (\delta_{ik}\delta_{jl}-\frac{1}{N}\delta_{jl})(\sigma_l^2+\epsilon)^{-1/2}-\frac{1}{2}(x_{kl}-\mu_l)\frac{d\sigma_l^2}{dx_{ij}}(\sigma_l^2+\epsilon)^{-3/2}
\end{eqnarray}
where 
$$\sigma_l^2 = \frac{1}{N}\sum_p \left(x_{pl}-\mu_l\right)^2$$
and then,
\begin{eqnarray}
\frac{d\sigma_l^2}{dx_{ij}} &=& \frac{1}{N}\sum_p2\left(\delta_{ip}\delta_{jl}-\frac{1}{N}\delta_{jl}\right)\left(x_{pl}-\mu_l\right)\\
&=&\frac{2}{N}(x_{il}-\mu_l)\delta_{jl}-\frac{2}{N^2}\sum_p\delta_{jl}\left(x_{pl}-\mu_l\right)\\
&=& \frac{2}{N}(x_{il}-\mu_l)\delta_{jl}
\end{eqnarray}

Putting everything together we thus have
\begin{eqnarray}
\frac{d\hat{x}_{kl}}{dx_{ij}} = (\delta_{ik}\delta_{jl}-\frac{1}{N}\delta_{jl})(\sigma_l^2+\epsilon)^{-1/2}-\frac{1}{N}(x_{kl}-\mu_l)(x_{il}-\mu_l)\delta_{jl}(\sigma_l^2+\epsilon)^{-3/2}
\end{eqnarray}

and therefore

\begin{eqnarray}
\frac{dL}{dx_{ij}} &=& \sum_{kl}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\hat{x}_{kl}}\frac{d\hat{x}_{kl}}{dx_{ij}}\\
&=&\sum_{kl}\frac{dL}{dy_{kl}}\gamma_l\delta_{jl}\left[(\delta_{ik}-\frac{1}{N})(\sigma_l^2+\epsilon)^{-1/2}-\frac{1}{N}(x_{kl}-\mu_l)(x_{il}-\mu_l)(\sigma_l^2+\epsilon)^{-3/2}\right]\\
&=&\sum_{k}\frac{dL}{dy_{kj}}\gamma_j\left[(\delta_{ik}-\frac{1}{N})(\sigma_j^2+\epsilon)^{-1/2}\right]-\sum_{k}\frac{dL}{dy_{kj}}\gamma_j\left[\frac{1}{N}(x_{kj}-\mu_j)(x_{ij}-\mu_j)(\sigma_j^2+\epsilon)^{-3/2}\right]\\
&=&\frac{dL}{dy_{ij}}\gamma_j(\sigma_j^2+\epsilon)^{-1/2}-\frac{1}{N}\sum_{k}\frac{dL}{dy_{kj}}\gamma_j(\sigma_j^2+\epsilon)^{-1/2}\\
& &-\frac{1}{N}\sum_{k}\frac{dL}{dy_{kj}}\gamma_j\left[(x_{kj}-\mu_j)(x_{ij}-\mu_j)(\sigma_j^2+\epsilon)^{-3/2}\right]\\
&=&\frac{1}{N}\gamma_j(\sigma_j^2+\epsilon)^{-1/2}\left[N\frac{dL}{dy_{ij}}-\sum_k\frac{dL}{dy_{kj}}-(x_{ij}-\mu_j)(\sigma_j^2+\epsilon)^{-1}\sum_k\frac{dL}{dy_{kj}}(x_{kj}-\mu_j)\right]
\end{eqnarray}


### Python implementation

Here is the simple python code for Batch-Normalization:

```python
# forward propagation part
xhat = (x - mu) / np.sqrt(var + eps)
y = gamma * xhat + beta

# backward part
dgamma = np.sum(dy * xhat, axis=0)
dbeta = np.sum(dy, axis=0)
dx = (dy - np.mean(dy, axis=0) - (x - mu) / (var + eps) * np.mean(dy * (x - mu), axis=0))
   * gamma / np.sqrt(var + eps)
```

### Issue about how `epsilon` is defined

If $\epsilon$ is defined the other way to avoid deviding by zero:

$$\hat{x_{kl}}=(x_{kl}-\mu_l)(\sigma_l+\epsilon)^{-1}$$

Then, for the back propagation part, `dx` should be revised as follows:

```python
# backward part
std = np.sqrt(var)
dx = (dy - np.mean(dy, axis=0) - (x - mu) / (std * (std + eps)) 
   * np.mean(dy * (x - mu), axis=0)) * gamma / np.sqrt(std + eps)
```


