---
layout: post
comments: true
title:  "Evolutionary algorithms"
excerpt: "Oswald Berthold - Quick introduction to evolutionary algorithms"
date:   2017-10-10
mathjax: true
hard_wrap: false
---

## Table of Contents ##

1.  [Overview](#org6c290b6)
2.  [Core algorithm](#orgb64c042)
3.  [Filling in the details](#org787a0f6)
4.  [Main types of evolutionary algorithms](#org42e8314)
5.  [Advanced techniques](#org6be0c43)
6.  [Examples](#org0e5d161)
7.  [Subfields](#org1d8ba56)
8.  [Relations to other fields](#orgfce7655)
9.  [References](#orgfd68185)

<span style="color:red">WARNING: I will have to do the graphics on the board.</span>  


<a id="org6c290b6"></a>

## Overview ##

Evolutionary algorithms (EAs) are a family of bioinspired algorithms <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> that generalize various techniques invented some time between 1930 and the 1960s. EAs have been in widespread use in science, technology, and art since, and are still considered state of the art in many problems in optimization, machine learning, robotics, or artificial life. There is a large community driving to expand the field with >6000 authors counted in 2005 \cite{Cotta2007}. One major conference is GECCO - The Genetic and Evolutionary Computation Conference <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>, with 13 tracks and 180 full papers in 2017.  

Why do we want to model natural evolution? Because obviously it is a powerful search technique. We would like to replicate this functionality and apply it to our own difficult search and optimization problems.  


<a id="orgb64c042"></a>

## Core algorithm ##

In (Floreano & Mattiussi, 2008) the four pillars of evolutionary theory are *population*, *diversity*, *heredity*, and *selection*. From this, a simple mathematical model of natural evolution is a tuple  

$$ \begin{equation} \text{EA} = (e, p, f, o, s) \end{equation} $$  

with environment $e$, population $p$, fitness $f$, operators $o$ and some generation statistics $g$. A minimal algorithm for improving the performance of $p$ with respect to environment and fitness $f(p, e)$ based on the model (1) looks like this:  

    1 evolutionary_algorithm:
    2   initialize e, p, f, o, s, p_
    3   do min(maximum iterations, forever)
    4     for each individual i in p
    5       compute fitness f_i of i in e
    6     for each slot i_ in p_
    7       randomly select individuals i,j from p with probability weighted by fitness 
    8       apply operators o to pair and store results in slot
    9     store entire state in s
    10    update p <- p_

Let's step through the algorithm line by line. Line 2 initializes the components, with p\_ some temporary memory. Line 3 then iterates the following block for a maximum number of iterations. In line 5 an individual is evaluated in the environment and its fitness computed, which is done for each member of the population in line 4. Line 6 creates a new population by sampling pairs of last generation individuals weighted by their fitness in line 7 and computing a new individual from the pair using the genetic operators in line 8. Line 9 updates the overall statistics and line 10 replaces the old population with a new one and jumps back to line 4.  


<a id="org787a0f6"></a>

## Filling in the details ##

Why and how should this work? Let's have a look at it in more detail.  

The *population* is a list of individuals and its size is another important parameter controlling diversity and convergence. *Individuals* represent candidate solutions to the problem we would like to solve. They usually consist of a *genotype* and a *phenotype*. The genotype can be a string of bits, reals, or a more complex structure like a nested list or a program trees.  

Usually, the fitness cannot be evaluated directly on the genotype but is translated first into a corresponding phenotype. The phenotype is then put into the environment of the problem and is evaluated by being allowed an episode of 'doing its thing'.  

The genotypic encoding and the mapping from genotype to phenotype are important parameters of an EA. For example when optimizing a function, the genotype can directly encode the argument at which the function should be evaluated. When tuning a controller in a dynamic simulation, the genotype might just encode the controller's parameters. For evaluation the parameterized controller needs to be simulated on the system for a given amount of time in order to compute a meaningful fitness.  


### Environment ###

The environment takes an individual $p_i\text{'s}$ phenotype as an input and outputs an episode of data corresponding to that individual. In many cases the returned data will be randomly distributed conditioned on the phenotype input, due to domain complexity, which has to be taken into account be the algorithm. Very similar to policy search.  


### Fitness ###

A fitness function $f$ is a function mapping from the combined space of individuals and their data episodes to the reals, e.g. $f(p_i, x) \rightarrow \mathbb{R}$. The function encodes, how an individual's fitness with respect to some desired configuration or behavior is computed from the data returned by the environment.  

> &#x2026; while in natural evolution the fitness of an individual  
> is defined by its reproductive success (number of offspring), in artificial evo-  
> lution the fitness of an individual is a function that measures how well that  
> individual solves a predefined problem. (Floreano & Mattiussi, 2008, pg. 1)  


### Genetic operators ###

The operators $o$ are stochastic transformations mapping from the space of $n$-tuples of individuals to the space of individuals. The operators produce new solutions from existing ones.  

The three main operators are *selection*, *mutation*, and *crossover*, also known as recombination. Genetic operators take one or more genomes as input and return a single new genome. The simplest type of EA results from using only selection and mutation. Selection retains a memory of solutions and their fitness, and mutation provides the necessary exploration around the fittest candidates. This work well in many situations but including crossover in between selection and mutation can a) accelerate the search (the real benefit of sexual reproduction) and b) enable *jump* access to regions in configuration space which are unreachable using mutation alone due to local minima. Mutation and combination are local and global exploration functions respectively. Both a) and b) depend on appropriate genotype to phenotype mapping.  


### Hyperparameters ###

Even a simple EA quickly accumulates many hyperparameters like population size, genetic representation, geno-pheno mapping, mutation rate, combination rules, mixing coefficients in composite fitness functions, and so on.  

The main tools for choosing hyperparameters are defaults, empirical estimates, first principles, or optimization. For example, the statistics $s$ can be used to modulate hyperparameters like mutation rate, recombination rules, or fitness weights. In addition, $s$ is the basis for graphical analyses of the experiment.  


<a id="org42e8314"></a>

## Main types of evolutionary algorithms ##

There are a few basic types of EAs which are distinguished by the type of genome they use.  

*Genetic algorithms* use strings of bits as the basis of the genetic encoding. The bits could correspond to pixel in a bitmap, inclusion of options in a configuration, or turns to make traversing a binary tree.  

*Evolution strategies* are based on real-valued vectors of parameters owing to parametric design methods of hardware evolution.  

*Genetic programming* is slightly different and includes an implicit layer of a developmental encoding. In GP the genome encodes a computation graph and the algorithm explores the space of a family of programs which can bump the expressive power of the genome by oom.  


<a id="org6be0c43"></a>

## Advanced techniques ##

-   CMA-ES
-   Coevolution
-   Open-ended evolution
-   Diversity
-   Developmental encodings: CPPNs, NEAT, HyperNEAT, map-elites, &#x2026;
-   Modularity
-   Distributed representation and genetic drift
-   Genetic regulatory pathways


<a id="org0e5d161"></a>

## Examples ##

-   pathnet, fernando (!!!)
-   blind watchmaker, Dawkins
-   creatures, Karl Sims
-   picbreeder, Stanley
-   evolving compressed weights, Koutnik et al.
-   robot motion recovery, Mouret Nature '16
-   Evolution of soft robots, Josh Bongard
-   compensating motion parallax, Pfeifer
-   NASA Antenna
-   FPGA circuit evolution, Thompson
-   FPGA evolved radio-receiver, ?
-   Modularity (Clune, Mouret & Lipson), (de Nardi, Holland et al.)
-   Self-replicating robot, Fumiya Iida
-   Evolved lego bridge, Bonabeau & &#x2026;
-   Fahrende Platine, Ferry

-   Neuroevolution, evolutionary optimization of quadrotor PID  
    controller parameters, intrinsic evolution of FPGA LUT configuration, complexity search / evoplast


<a id="org1d8ba56"></a>

## Subfields ##

-   Evolvable hardware, intrinsic evolution, in-silico evolution
-   Evolutionary robotics
-   Evol. parameter optimization, hyper-parameter optimization
-   Neuroevolution
-   Theory of evolution


<a id="orgfce7655"></a>

## Relations to other fields ##

Evolutionary methods are closely linked with several other families of computational methods. For example, EAs can be framed and understood in terms of stochastic optimization, black-box optimization, particle based methods, or policy search by PG or CACLA / EH aka "cling to the best you you've seen and search around there".  


<a id="orgfd68185"></a>

## References ##

-   David B. Fogel, 2000, Evolutionary computation
-   Mitchell & Tayler, 1999, Evolutionary Computation : An Overview
-   John Holland, 1975, Adaptation in natural and artificial systems
-   Ingo Rechenberg, 1973, Evolutionsstrategie
-   Nolfi & Floreano, 2000, Evolutionary robotics - The biology, intelligence, and technology of self-organizing machines
-   Floreano & Mattiussi, 2008, Bio-inspired artificial intelligence


## Footnotes ##

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Inspired by the theory of natural evolution.

<sup><a id="fn.2" href="#fnr.2">2</a></sup> Website of SIGEVO, the ACM Special Interest Group on Genetic and Evolutionary Computation <http://sig.sigevo.org/index.html/tiki-index.php#&panel1-1>
