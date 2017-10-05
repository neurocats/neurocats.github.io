
# Table of Contents

1.  [Introduction](#org7da000f)
    1.  [Preliminaries](#org686636b)
    2.  [Graphical models](#org1652d7c)
    3.  [Graphical models](#org45b0438)
2.  [Representation](#orgcf0fe5d)
    1.  [Graphical models: Representation](#org628490c)
    2.  [Graphical models: Representation](#org71ab20a)
    3.  [Graphical models: Bayesian networks](#org43814c0)
    4.  [Graphical models: Bayesian networks](#org9061324)
    5.  [Graphical models examples: PCA, ICA, &#x2026;](#orgbb5638d)
    6.  [Graphical models examples: PCA, ICA, &#x2026;](#org1b499a5)
    7.  [Graphical models examples: PCA, ICA, &#x2026;](#org79c3351)
    8.  [Graphical models examples: Temporal models](#org9f1638a)
    9.  [Graphical models examples: Temporal models](#org4002512)
    10. [Graphical models examples: Undirected graphs](#orged454a9)
    11. [Graphical models summary](#org81baad2)
3.  [Inference](#org50e630f)
    1.  [Graphical models: Inference](#org7ab6ac8)
    2.  [Graphical models: Inference](#org7bf516c)
4.  [Learning](#orgb7fc603)
    1.  [Graphical models: Learning](#org61e529d)
5.  [Action and decisions](#orgbebf2b8)
    1.  [Making decisions](#org0aac30e)
6.  [Realworld models](#orgcd0d282)
    1.  [Generative models](#orgec26ede)
    2.  [Boltzmann machine](#org0fe6f89)
    3.  [Dynamic bayesian networks](#org0b032dc)
    4.  [Variational autoencoder](#org9ba1800)
    5.  [Mixture of experts](#orge507284)
7.  [References](#org66607f8)
    1.  [Sources](#org82e71be)
    2.  [Online](#orge0cc074)
    3.  [Software environments](#orgaccc3cb)



<a id="org7da000f"></a>

# Introduction


<a id="org686636b"></a>

## Preliminaries

\mypara{Modelling}Building models is a main activity within science and the models
surviving the current battery of tests are the main output of any
scientific discipline.

\mypara{Machine learning}Mathematical and computational models

\mypara{Artificial intelligence}Bla &#x2026;


<a id="org1652d7c"></a>

## Graphical models

Graphical models are a marriage between probability theory and graph theory.  

Dealing with *uncertainty* and *complexity*

  

Tool for design and analysis of machine learning algorithms

  

Graphical part allows modularity (repetition of simple parts)  

Probabilistic part allows to connect modules among themselves and to
connect modules to data (inference, learning, &#x2026;).  


<a id="org45b0438"></a>

## Graphical models

GM's generalize many special case developed and used in statistics,
information theory, systems engineering, pattern recognition and
statistical mechanics.  

**Examples** include mixture models, factor analysis, hidden Markov
models, Kalman filters and Ising models.  

Topics: Representation, inference, learning, decision theory, applications


<a id="orgcf0fe5d"></a>

# Representation


<a id="org628490c"></a>

## Graphical models: Representation

A graphical model is a *graph* where the nodes represent random
variables.

Edges represent conditional dependence among variables, absence of
edges indicates independence.

Dependencies allow much smaller effective joint densities, \(N\) binary
variables full density needs \(\mathcal{O}(2^N)\) parameters, graphical
repr may need much less.


<a id="org71ab20a"></a>

## Graphical models: Representation

Two main types of graphical models: directed and undirected

-   fully undirected graphs: **Markov Random Fields**
-   fully directed and acyclic: **Bayesian networks**

(Actually three: mixed edge type is called *chain graph*)  

Bayesian networks are also known as belief network, *generative models*, causal
models, &#x2026;


<a id="org43814c0"></a>

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


<a id="org9061324"></a>

## Graphical models: Bayesian networks

**Example**

\begin{figure}
  \includegraphics[width=0.8\textwidth]{img/p_diss/graphical/gm_simple_bn_c.jpg}
  \caption{Grass wet example from Murphy02}
\end{figure}


<a id="orgbb5638d"></a>

## Graphical models examples: PCA, ICA, &#x2026;

Asking questions about \(P(Y,X) = P(X) P(Y|X)\) with \(Y\) observed and
\(X\) latent.

  

Factor analysis which has classical PCA as limit case.

  

Mixtures of factor analysis and independent factor analysis with IFA
relating to ICA.


<a id="org1b499a5"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="org79c3351"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="org9f1638a"></a>

## Graphical models examples: Temporal models

These are also called Dynamic Bayesian Networks (DBN) although dynamic
means temporal.  

Hidden Markov Model as a 2TBN

\[
P(Q, Y) = P(Q_1) P(Y_1 | Q_1) \Pi_{t=2}^4 P(Q_t | Q_{t-1}) P(Y_t | Q_t)
\]


<a id="org4002512"></a>

## Graphical models examples: Temporal models

Kalman filter

\[\begin{array}{rcl}
P(X_1 = x) & = & \mathcal{N}(x;x_0, V_0) \\
P(X_{t+1} = x_{t+1} | U_t = u, X_t = x) & = & \mathcal{N}(x_{t+1}; Ax + Bu, Q) \\
P(Y_{t+1} = y_{t+1} | X_t = x, U_t = u) & = & \mathcal{N}(y_{t+1}; Cx + Du, R) \\
\end{array}\]


<a id="orged454a9"></a>

## Graphical models examples: Undirected graphs

Markov Random Field, e.g. Ising model, graph is a homogenous grid


<a id="org81baad2"></a>

## Graphical models summary


<a id="org50e630f"></a>

# Inference


<a id="org7ab6ac8"></a>

## Graphical models: Inference

Estimate or predict the values of hidden variables given
observations.  

Observing leaves and predicting causes: bottom-up, inference, diagnosis, &#x2026;  

Observing roots and predicting consequences: top-down, prediction, &#x2026;\\

Exact inference: variable elimination (driving in independent sums),
dynamic programming. Running time potentially very bad, thus approximations.  


<a id="org7bf516c"></a>

## Graphical models: Inference

Approximate inference: sampling, variational techniques, loopy belief
propagation  

Sampling: importance sampling, sampling from prior p(x) and weighting
samples by posterior likelhood p(y|x).  

Variational inference: approximate true posterior P with simpler Q,
choosing Q minimizing the Kullback-Leibler divergence
\(D_{KL}(Q||P)\).  


<a id="orgb7fc603"></a>

# Learning


<a id="org61e529d"></a>

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


<a id="orgbebf2b8"></a>

# Action and decisions


<a id="org0aac30e"></a>

## Making decisions

Attach utility to variables as in “Decision Theory = Probability Theory + Utility Theory”


<a id="orgcd0d282"></a>

# Realworld models


<a id="orgec26ede"></a>

## Generative models

Representing variables probabilistically (even improperly) e.g. as
densities makes the model *generative*, the model can be sampled with
respect to prediction time conditioning.


<a id="org0fe6f89"></a>

## Boltzmann machine

-   steady state inputs
-   no temporal sequence order


<a id="org0b032dc"></a>

## Dynamic bayesian networks

Markov chains, Bayes filters, state space models and their
representation as a DBN

Consider: autoencoder/hidden space sequence modelling/autoencoder


<a id="org9ba1800"></a>

## Variational autoencoder


<a id="orge507284"></a>

## Mixture of experts

Gaussian experts: gaussian mixture density model

Pg. 7 Murphy 2001 Intro do graphical models / BNs


<a id="org66607f8"></a>

# References


<a id="org82e71be"></a>

## Sources

\scriptsize

1.  Main sources

    -   Murphy, 1998, A Brief Introduction to Graphical Models and Bayesian
        Networks, <https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html>
    -   Murphy, 2001, An introduction to graphical models,
        <https://www.cs.ubc.ca/~murphyk/Papers/intro_gm.pdf>
    -   Murphy, 2002, Dynamic Bayesian Networks: Representation, inference,
        learning,

2.  Additional sources

    -   Koller & Friedmann, 2009, Probabilistic graphical models, MIT Press
    -   Haykin, 2013, Neural networks and learning machines (3rd Ed.),
        Pearson Education
    -   Russell & Norvig, 2003, Artificial intelligence - A modern
        approach, Pearson Education


<a id="orge0cc074"></a>

## Online

\scriptsize

-   <https://en.wikipedia.org/wiki/Bayesian_network>
-   <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>


<a id="orgaccc3cb"></a>

## Software environments

\scriptsize

-   scikit naive bayes: <http://scikit-learn.org/stable/modules/naive_bayes.html>

