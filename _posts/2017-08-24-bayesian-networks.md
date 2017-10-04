---
layout: post
comments: true
title:  "Introduction to Bayesian networks Pt. 1"
excerpt: "Oswald Berthold - A smooth introduction to Bayesian networks"
date:   2017-08-24
mathjax: true
---

# Table of Contents

1.  [Introduction](#orgb6be35f)
    1.  [Graphical models](#orgaf6b925)
    2.  [Graphical models](#orgeabae00)
2.  [Representation](#orgf04fce7)
    1.  [Graphical models: Representation](#orga73eccb)
    2.  [Graphical models: Representation](#orgf74365f)
    3.  [Graphical models: Bayesian networks](#org7f17a30)
    4.  [Graphical models: Bayesian networks](#orgd3be1d1)
    5.  [Graphical models examples: PCA, ICA, &#x2026;](#orgfc8298f)
    6.  [Graphical models examples: PCA, ICA, &#x2026;](#orgb47cf57)
    7.  [Graphical models examples: PCA, ICA, &#x2026;](#org5eb7865)
    8.  [Graphical models examples: Temporal models](#orgeb8bb22)
    9.  [Graphical models examples: Temporal models](#orgf169532)
    10. [Graphical models examples: Undirected graphs](#org77cf048)
    11. [Graphical models summary](#org08fef05)
3.  [Inference](#org8545e20)
    1.  [Graphical models: Inference](#org9b922d7)
    2.  [Graphical models: Inference](#org3d2dcb1)
4.  [Learning](#org4c626f2)
    1.  [Graphical models: Learning](#orgef8c252)
5.  [Action and decisions](#org6dcdc41)
    1.  [Making decisions](#org55926e7)
6.  [Realworld models](#org58b2d46)
    1.  [Generative models](#orgac014f3)
    2.  [Boltzmann machine](#org56f94cb)
    3.  [Dynamic bayesian networks](#orgc268406)
    4.  [Variational autoencoder](#orgf65adb3)
    5.  [Mixture of experts](#orgd248cff)
7.  [References](#org391aba4)
    1.  [Sources](#org8118fbb)
    2.  [Online](#org0790a90)
    3.  [Software environments](#orgb464025)



<a id="orgb6be35f"></a>

# Introduction


<a id="orgaf6b925"></a>

## Graphical models

(Basically all Murphy98/01/02 material)   

A marriage between probability theory and graph theory  

Dealing with *uncertainty* and *complexity*

  

Tool for design and analysis of machine learning algorithms

  

Graphical part allows modularity (repetition of simple parts)  

Probabilistic part allows to connect modules among themselves and to
connect modules to data (inference, learning, &#x2026;).  


<a id="orgeabae00"></a>

## Graphical models

GM's generalize many special case developed and used in statistics,
information theory, systems engineering, pattern recognition and
statistical mechanics.  

**Examples** include mixture models, factor analysis, hidden Markov
models, Kalman filters and Ising models.  

Topics: Representation, inference, learning, decision theory, applications


<a id="orgf04fce7"></a>

# Representation


<a id="orga73eccb"></a>

## Graphical models: Representation

A graphical model is a *graph* where the nodes represent random
variables.

Edges represent conditional dependence among variables, absence of
edges indicates independence.

Dependencies allow much smaller effective joint densities, \(N\) binary
variables full density needs \(\mathcal{O}(2^N)\) parameters, graphical
repr may need much less.


<a id="orgf74365f"></a>

## Graphical models: Representation

Two main types of graphical models: directed and undirected

-   fully undirected graphs: **Markov Random Fields**
-   fully directed and acyclic: **Bayesian networks**

(Actually three: mixed edge type is called *chain graph*)  

Bayesian networks are also known as belief network, *generative models*, causal
models, &#x2026;


<a id="org7f17a30"></a>

## Graphical models: Bayesian networks

Directed acyclic graph, not inherently Bayesian but *systematically* using Bayes rule
for inference.

1.  Bayes rule

    \[\begin{array}{rcllll}
    \text{posterior} & & \text{likelihood} & \text{prior} & & \text{marginal likelihood} \\
    p(y|x) & = & p(x|y) & p(y) & / & p(x) \\
    p(\theta|x) & = & p(x|\theta) & p(\theta) & / & p(x) \\
    \end{array}\]
    
    with posterior \(y\) and data \(x\) or parameters \(\theta\) and data \(x\)


<a id="orgd3be1d1"></a>

## Graphical models: Bayesian networks

**Example**

\begin{figure}
  \includegraphics[width=0.8\textwidth]{img/p_diss/graphical/gm_simple_bn_c.jpg}
  \caption{Grass wet example from Murphy02}
\end{figure}


<a id="orgfc8298f"></a>

## Graphical models examples: PCA, ICA, &#x2026;

Asking questions about \(P(Y,X) = P(X) P(Y|X)\) with \(Y\) observed and
\(X\) latent.

  

Factor analysis which has classical PCA as limit case.

  

Mixtures of factor analysis and independent factor analysis with IFA
relating to ICA.


<a id="orgb47cf57"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="org5eb7865"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="orgeb8bb22"></a>

## Graphical models examples: Temporal models

These are also called Dynamic Bayesian Networks (DBN) although dynamic
means temporal.  

Hidden Markov Model as a 2TBN

\[
P(Q, Y) = P(Q_1) P(Y_1 | Q_1) \Pi_{t=2}^4 P(Q_t | Q_{t-1}) P(Y_t | Q_t)
\]


<a id="orgf169532"></a>

## Graphical models examples: Temporal models

Kalman filter

\[\begin{array}{rcl}
P(X_1 = x) & = & \mathcal{N}(x;x_0, V_0) \\
P(X_{t+1} = x_{t+1} | U_t = u, X_t = x) & = & \mathcal{N}(x_{t+1}; Ax + Bu, Q) \\
P(Y_{t+1} = y_{t+1} | X_t = x, U_t = u) & = & \mathcal{N}(y_{t+1}; Cx + Du, R) \\
\end{array}\]


<a id="org77cf048"></a>

## Graphical models examples: Undirected graphs

Markov Random Field, e.g. Ising model, graph is a homogenous grid


<a id="org08fef05"></a>

## Graphical models summary


<a id="org8545e20"></a>

# Inference


<a id="org9b922d7"></a>

## Graphical models: Inference

Estimate or predict the values of hidden variables given
observations.  

Observing leaves and predicting causes: bottom-up, inference, diagnosis, &#x2026;  

Observing roots and predicting consequences: top-down, prediction, &#x2026;\\

Exact inference: variable elimination (driving in independent sums),
dynamic programming. Running time potentially very bad, thus approximations.  


<a id="org3d2dcb1"></a>

## Graphical models: Inference

Approximate inference: sampling, variational techniques, loopy belief
propagation  

Sampling: importance sampling, sampling from prior p(x) and weighting
samples by posterior likelhood p(y|x).  

Variational inference: approximate true posterior P with simpler Q,
choosing Q minimizing the Kullback-Leibler divergence
\(D_{KL}(Q||P)\).  


<a id="org4c626f2"></a>

# Learning


<a id="orgef8c252"></a>

## Graphical models: Learning

Huge topic in itself, learning *structure* or *parameters* or both.  

Four case:

\begin{tabular}{c|cc}
 & & Observability \\
Structure & Full & Partial \\
\hline
known & Closed form & EM \\
unknown & Local search & Structural EM \\
\end{tabular}

  

Known structure - full observability; Known structure - partial observability; Unknown structure - full observability; Unknown structure - partial observability (POMDP)  


<a id="org6dcdc41"></a>

# Action and decisions


<a id="org55926e7"></a>

## Making decisions

Attach utility to variables as in “Decision Theory = Probability Theory + Utility Theory”


<a id="org58b2d46"></a>

# Realworld models


<a id="orgac014f3"></a>

## Generative models

Representing variables probabilistically (even improperly) e.g. as
densities makes the model *generative*, the model can be sampled with
respect to prediction time conditioning.


<a id="org56f94cb"></a>

## Boltzmann machine

-   steady state inputs
-   no temporal sequence order


<a id="orgc268406"></a>

## Dynamic bayesian networks

Markov chains, Bayes filters, state space models and their
representation as a DBN

Consider: autoencoder/hidden space sequence modelling/autoencoder


<a id="orgf65adb3"></a>

## Variational autoencoder


<a id="orgd248cff"></a>

## Mixture of experts

Gaussian experts: gaussian mixture density model

Pg. 7 Murphy 2001 Intro do graphical models / BNs


<a id="org391aba4"></a>

# References


<a id="org8118fbb"></a>

## Sources

\scriptsize

-   Murphy, 1998, A Brief Introduction to Graphical Models and Bayesian
    Networks, <https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html>
-   Murphy, 2001, An introduction to graphical models,
    <https://www.cs.ubc.ca/~murphyk/Papers/intro_gm.pdf>
-   Murphy, 2002, Dynamic Bayesian Networks: Representation, inference,
    learning,

-   Koller & Friedmann, 2009, Probabilistic graphical models, MIT Press
-   Haykin, 2013, Neural networks and learning machines (3rd Ed.),
    Pearson Education
-   Russell & Norvig, 2003, Artificial intelligence - A modern
    approach, Pearson Education


<a id="org0790a90"></a>

## Online

\scriptsize

-   <https://en.wikipedia.org/wiki/Bayesian_network>
-   <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>


<a id="orgb464025"></a>

## Software environments

\scriptsize

-   scikit naive bayes: <http://scikit-learn.org/stable/modules/naive_bayes.html>

