---
layout: post
comments: true
title:  "Introduction to Bayesian networks Pt. 1"
excerpt: "Oswald Berthold - A smooth introduction to Bayesian networks"
date:   2017-08-24
mathjax: true
input: GFM
hard_wrap: false
---
<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#org51b5e2a">1. Preamble</a></li>
<li><a href="#orgb74a732">2. Introduction</a></li>
<li><a href="#org85a01fb">3. Representation</a></li>
<li><a href="#org7ccf05f">4. Inference</a></li>
<li><a href="#org6f50ab6">5. Learning</a></li>
<li><a href="#org31afab1">6. Action and decisions</a></li>
<li><a href="#org341c82d">7. Realworld models</a></li>
<li><a href="#orgc8d83d9">8. References</a></li>
</ul>
</div>
</div>
<span style="color:red">WARNING: Work in progress / Converting presentation slides to full text</span>  


<a id="org51b5e2a"></a>

## Preamble ##

<span style="color:orange">TODO: Move this section to separate mini-article</span>  

<span style="color:red">WARNING: Philosophy</span>  


<a id="org144e7ce"></a>

### Models ###

The natural world can be seen as composed of different processes like  
planets orbiting stars, the melting of a glacier, or people. Processes  
communicate via *observables* which are projections of a process's  
internal state into some measurement space. The measurement space is  
in general the input space of another process. Usually these  
projection maps both mix and destroy information and measurements bear  
less information than its original and true *cause*. Nonetheless it is  
possible in many cases to reconstruct something close to the original  
cause by mapping from a series of observations back to the space of  
the candidate process's internal state. Such maps are called *models*  
of a corresponding process.  


<a id="org8fb16d5"></a>

### Models, explanation and agents ###

Building models of processes is the purpose of science. The purpose of  
a model is to explain observations by inferring a process and  
corresponding hidden state which is agreement with the observations,  
and to predict future observations from exploration of the hidden  
space. Curiously, model building is also thought to be taking place in  
humans and other animals at conscious and non-conscious levels (1),  
and, by functional transfer, can be used to build sophisticated  
artificial agents (also known as brains) as networks of interacting  
models (2). The motivation for writing this article is to help  
understand functions known from (1) and to enable transfer to (2).  


<a id="org278d253"></a>

### Autonomous modelling ###

The difference between science and artificial agents is that in  
science *people* are building the models, using mostly high-level  
brain functions whereas (anything close to) a sophisticated agent has  
to build the models itself. An agent that is building a model of its  
observations (data) is a learning agent. Technically, many proposed  
agents are *fitting* a template model to best explain the observed  
data. Fitting basically means selecting a model from a large family of  
models. The model is selected using parameters, usually called  
\[\theta\]. Many methods of model fitting are known in the  
literature. The hypothesis is, that since all of these methods solve  
very similar problems, they are likely to have common issues. We look  
at different approaches to be able to compare the methods and transfer  
results obtained for one method to other ones.  


<a id="orgb74a732"></a>

## Introduction ##


<a id="org55efda3"></a>

### Graphical models ###

This article gives a quick introduction to *Bayesian networks* and  
their use in statistics, machine learning and artificial  
intelligence. Bayesian networks turn out to be better described as  
probabilistic *graphical models*, which is the more general term and  
is adopted here. Graphical models (GM) are a marriage between  
*probability theory* and *graph theory*. The probabilistic part is  
used to represent *uncertainties* and the graphical part allows to  
cope with *complexity*.  

The graphical approach is a powerful tool for the design and analysis  
of machine learning algorithms in general. Here we are mostly  
concerned with *learning* models from data, which is also referred to  
as *fitting* a model to *data*. Real world data is inherently  
incomplete due to partial observability, and uncertain due to  
incompleteness and noise, which is a fact that machine learning  
representations and algorithms have to take care of. Probabilistic  
aproaches and particularly GMs can do this quite elegantly.  

Graphical models emerged as a generalization of many special cases  
developed and used historically in statistics, information theory,  
systems engineering, pattern recognition and mathematical  
physics. Some well known examples include mixture models, factor  
analysis, hidden Markov models, Bayes filters, or Ising models.  

The remaining text is first going to finish the introducton with some  
additional terminology. This is followed by a section of each of the  
topics of representation, inference, learning, decision theory and  
concluded with a discussion of use-cases and applications and a  
final summary.  


<a id="org85a01fb"></a>

## Representation ##


<a id="org46cb2ad"></a>

### Nodes and edges ###

A graphical model is a graph $G = (V, E)$ whose nodes $v_i \in V, i =
 1, ..., |G|$ represent random variables. The graph's edges $e_j \in
 E$ represent conditional dependences among variables. The absence of  
an edge indicates conditional independence. A modelling problem would  
be solved if one knows, or has learned from observations, the joint  
density of all the model's variables. The storage requirements of the  
full joint density representation are of exponential order  
$\mathcal{O}^N$ in the number of variables $N$ <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>, which is a  
problem. That problem is repaired by the *manifold*  
hypothesis <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> which says that high-dimensional data tends to be  
narrowly distributed on lower-dimensional manifolds. The manifold is  
said to be *embedded* in the full space.  

In probabilistic terms this property of sparse distribution is  
reflected as conditional dependencies. Dependencies indicate the  
presence of information which is *shared* among the pair of  
variables and which implies compressibility. For example two variables  
might share a global property like location but differ in local  
properties like color. In graphical models dependencies are exploited  
by constructing a compressed effective joint density using  
significantly less edges than the fully connected graph.  

There are two types of edges in vanilla graphs, directed ones and  
undirected ones. Graphical models using only undirected edges are  
called *Markov Random Fields*, and cycle free models using only  
directed edges are called *Bayesian networks*. Graphs with mixed edge  
types are called *chain graphs*. Historically, Bayesian networks have  
also been known as belief networks, *generative models*, or causal  
models.  


<a id="org86f2d48"></a>

## Graphical models: Bayesian networks

Directed acyclic graph, not inherently Bayesian but *systematically* using Bayes rule  
for inference.  


<a id="orgf76d4a0"></a>

#### Bayes rule ####

\[\begin{array}{rcllll}
 \text{posterior} & & \text{likelihood} & \text{prior} & & \text{marginal likelihood} \\
 p(y|x) & = & p(x|y) & p(y) & / & p(x) \\
 p(\theta|x) & = & p(x|\theta) & p(\theta) & / & p(x) \\
 \end{array}\]  

with posterior $y$ and data $x$ or parameters $\theta$ and data $x$  


<a id="orgbf5f581"></a>

## Graphical models: Bayesian networks

**Example**  

\begin{figure}
  \includegraphics[width=0.8\textwidth]{img/p_diss/graphical/gm_simple_bn_c.jpg}
  \caption{Grass wet example from Murphy02}
\end{figure}


<a id="org912fc50"></a>

## Graphical models examples: PCA, ICA, &#x2026;

Asking questions about $P(Y,X) = P(X) P(Y|X)$ with $Y$ observed and  
$X$ latent.  

#   #

Factor analysis which has classical PCA as limit case.  

#   #

Mixtures of factor analysis and independent factor analysis with IFA  
relating to ICA.  


<a id="orgaabeaeb"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="orgfc6cae7"></a>

## Graphical models examples: PCA, ICA, &#x2026;


<a id="org8c76fc0"></a>

## Graphical models examples: Temporal models

These are also called Dynamic Bayesian Networks (DBN) although dynamic  
means temporal.  

Hidden Markov Model as a 2TBN  

\[
 P(Q, Y) = P(Q_1) P(Y_1 | Q_1) \Pi_{t=2}^4 P(Q_t | Q_{t-1}) P(Y_t | Q_t)
 \]  


<a id="orga7b6746"></a>

## Graphical models examples: Temporal models

Kalman filter  

\[\begin{array}{rcl}
 P(X_1 = x) & = & \mathcal{N}(x;x_0, V_0) \\
 P(X_{t+1} = x_{t+1} | U_t = u, X_t = x) & = & \mathcal{N}(x_{t+1}; Ax + Bu, Q) \\
 P(Y_{t+1} = y_{t+1} | X_t = x, U_t = u) & = & \mathcal{N}(y_{t+1}; Cx + Du, R) \\
 \end{array}\]  


<a id="org89530e5"></a>

## Graphical models examples: Undirected graphs

Markov Random Field, e.g. Ising model, graph is a homogenous grid  


<a id="org33a5b58"></a>

### Graphical models summary ###


<a id="org7ccf05f"></a>

## Inference ##


<a id="org3117a78"></a>

## Graphical models: Inference

Estimate or predict the values of hidden variables given  
observations.  

Observing leaves and predicting causes: bottom-up, inference, diagnosis, &#x2026;  

Observing roots and predicting consequences: top-down, prediction, &#x2026;\\  

Exact inference: variable elimination (driving in independent sums),  
dynamic programming. Running time potentially very bad, thus approximations.  


<a id="orgbe3a815"></a>

## Graphical models: Inference

Approximate inference: sampling, variational techniques, loopy belief  
propagation  

Sampling: importance sampling, sampling from prior p(x) and weighting  
samples by posterior likelhood p(y|x).  

Variational inference: approximate true posterior P with simpler Q,  
choosing Q minimizing the Kullback-Leibler divergence  
$D_{KL}(Q||P)$.  


<a id="org6f50ab6"></a>

## Learning ##


<a id="org17bc0b3"></a>

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

#   #

Known structure - full observability; Known structure - partial observability; Unknown structure - full observability; Unknown structure - partial observability (POMDP)  


<a id="org31afab1"></a>

## Action and decisions ##


<a id="org144d168"></a>

### Making decisions ###

Attach utility to variables as in “Decision Theory = Probability Theory + Utility Theory”  


<a id="org341c82d"></a>

## Realworld models ##


<a id="org39ec0a4"></a>

### Generative models ###

Representing variables probabilistically (even improperly) e.g. as  
densities makes the model *generative*, the model can be sampled with  
respect to prediction time conditioning.  


<a id="org2d31a4a"></a>

### Boltzmann machine ###

-   steady state inputs
-   no temporal sequence order


<a id="org057a89d"></a>

### Dynamic bayesian networks ###

Markov chains, Bayes filters, state space models and their  
representation as a DBN  

Consider: autoencoder/hidden space sequence modelling/autoencoder  


<a id="org8ca6a70"></a>

### Variational autoencoder ###


<a id="orge508a3b"></a>

### Mixture of experts ###

Gaussian experts: gaussian mixture density model  

Pg. 7 Murphy 2001 Intro do graphical models / BNs  


<a id="orgc8d83d9"></a>

## References ##


<a id="org165d4b3"></a>

### Sources ###



<a id="orgda11ee6"></a>

#### Main sources ####

-   Murphy, 1998, A Brief Introduction to Graphical Models and Bayesian  
    Networks, <https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html>
-   Murphy, 2001, An introduction to graphical models,  
    <https://www.cs.ubc.ca/~murphyk/Papers/intro_gm.pdf>
-   Murphy, 2002, Dynamic Bayesian Networks: Representation, inference,  
#    learning, #


<a id="orgd1b1585"></a>

#### Additional sources ####

-   Koller & Friedmann, 2009, Probabilistic graphical models, MIT Press
-   Haykin, 2013, Neural networks and learning machines (3rd Ed.),  
#    Pearson Education #
-   Russell & Norvig, 2003, Artificial intelligence - A modern  
#    approach, Pearson Education #


<a id="org8044cf3"></a>

### Online ###

-   <https://en.wikipedia.org/wiki/Bayesian_network>
-   <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>


<a id="orge1f0691"></a>

### Software environments ###

-   scikit naive bayes: <http://scikit-learn.org/stable/modules/naive_bayes.html>


## Footnotes ##

<sup><a id="fn.1" href="#fnr.1">1</a></sup> For $N$ binary variables the full density needs  
$\mathcal{O}(2^N)$ parameters. A dependency based set of edges can  
reduce the order significantly.

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <https://heavytailed.wordpress.com/2012/11/03/manifold-hypothesis-part-1-compression-learning-and-the-strong-mh/>  
on <span class="timestamp-wrapper"><span class="timestamp">&lt;2017-10-05 Do&gt;</span></span>
