# :high_brightness: Introduction to Simulation Based Inference: Enhancing Synthetic Models with Artificial Intelligence 
## JSC training course
![](data/Logo_FZJ_mit_jsc_1200px.jpg)

### Organizers
[Alina Bazarova](https://www.fz-juelich.de/profile/bazarova_al), [Jose Robledo](https://www.fz-juelich.de/profile/robledo_j)

### Information

- **Date**: September 7-8, 2026, 13:00-17:00
- **Place**: Online
- [Link to course page](https://www.fz-juelich.de/en/jsc/news/events/training-courses/training-courses-2026/simulation-base-inference)

### Course content 
This tutorial introduces Simulation-Based Inference (SBI), a framework combining Bayesian modeling, AI techniques, and high-performance computing (HPC) to address key challenges, such as performing reliable inference with limited data by using AI-based approximate Bayesian computation. Moreover, it tackles the problem of intractable likelihood functions, thereby allowing to utilize Bayesian inference for biological systems with multiple sources of stochasticity. The tutorial also demonstrates how to leverage HPC environments to drastically reduce inference runtimes, making it highly relevant for large-scale biological problems. This tutorial bridges theoretical foundations with hands-on applications realized via jupyter notebooks.

### Description
In the world of the research fields drifting further apart from one another and Artificial Intelligence (AI) tools gaining increasingly more attention, methods which can bring a number of seemingly disjoint fields together is of the utmost importance. The proposed tutorial is sought to provide researchers with an instrument to unify Bayesian modelling, large-scale simulations, and AI methods while integrating them into the HPC environment.

While Bayesian inference is widely used in the research community, as it provides distributional estimates of model parameters and allows to update the model by incorporating new data into it, it often suffers from computationally intensive processes and limited parallelization capabilities. Simulation Based Inference (SBI) is a tool to tackle this issue.

SBI employs AI-based approximate Bayesian computation to dramatically reduce inference times and generate reliable estimates, even when observed data are sparse. This approach enables any representative simulation model to inform parameter constraints, yielding approximate posterior distributions. Additionally, SBI facilitates workload distribution across high-performance computing clusters, further reducing runtime.

This tutorial discusses theoretical foundations and provides practical training for constructing SBI frameworks tailored to specific models. Through provided examples, participants will gain insight into various levels of model granularity, ranging from a simple black box approach to a highly customizable design, and develop the skills to effectively manage HPC devices within a given set-up. By participating in this tutorial, attendees will gain an understanding of the principles of Simulation Based Inference, learn how to apply this methodology in the context of HPC in a variety of case scenarios, to evaluate its potential and utility, and be encouraged to consider its applicability to their own research projects.

You may find the tutorial notebooks in the [notebooks](./notebooks) folder.

### Learning Objectives

- Understand the Principles of Simulation-Based Inference (SBI): learn the theoretical foundations of SBI, including its relationship with Bayesian inference and its advantages in handling complex biological systems.
- Explore SBI Methods (SNPE, SNLE, and SNRE): gain an understanding of Sequential Neural Posterior Estimation (SNPE), Sequential Neural Likelihood Estimation (SNLE), and Sequential Neural Ratio Estimation (SNRE) and their applications in computational biology.
- Learn how to design and implement SBI frameworks for representative biological scenarios, such as molecular dynamics, cell growth, count data modeling, and Lotka-Volterra systems.
- Leverage HPC for SBI Workflows: understand how to use high-performance computing (HPC) environments to scale SBI workflows and efficiently distribute computational workloads.

### Course schedule


### :books: References

1. The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 - [Link to paper :newspaper:](https://doi.org/10.1073/pnas.191278911)

2. Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016 - [Link to paper :newspaper:](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)

3. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray - [Link to paper :newspaper:](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)

4. Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning - [Link to paper :newspaper:](http://proceedings.mlr.press/v119/hermans20a.html)

5. Simulation-Based Inference: A Practical Guide, Michael Deistler, Jan Boelts, Peter Steinbach, Guy Moss, Thomas Moreau, Manuel Gloeckler, Pedro L. C. Rodrigues, Julia Linhart, Janne K. Lappalainen, Benjamin Kurt Miller, Pedro J. Gonçalves, Jan-Matthis Lueckmann, Cornelius Schröder, Jakob H. Macke, 2025 - [Link to paper](https://arxiv.org/abs/2508.12939)





