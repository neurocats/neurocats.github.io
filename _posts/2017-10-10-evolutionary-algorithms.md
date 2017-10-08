---
layout: post
comments: true
title:  "Introduction to evolutionary algorithms"
excerpt: "Oswald Berthold - Brief introduction to evolutionary algorithms"
date:   2017-10-10
mathjax: true
input: GFM
hard_wrap: false
---

## Table of Contents ##

1.  [Evolutionary algorithms](#org366d6af)



<a id="org366d6af"></a>

## Evolutionary algorithms ##

A mathematical model of the most basic mechanisms of natural evolution  
is a tuple $E = (e, p, f, o, s)$ with environment $e$, population $p$,  
fitness $f$, operators $o$ and some generation statistics $g$.  

Why do we want to model evolution? Because we want to optimize complex  
functions and evolutionary models can do that.  

An algorithm for doing this  

\#+BEGIN<sub>EXAMPLE</sub>  
evolution  
  popsize = |p|  
#  do forever   #
#    for each individual in p   #
#      compute fitness   #
    for each individual in p+  
#      randomly select 2 individuals from p with fitness weighted probability   #
#      apply all operators in o to pair   #
    assign p <- p+  
\\#+END<sub>EXPORT</sub>  

