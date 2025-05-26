# :high_brightness: Introduction to Simulation Based Inference: enhancing synthetic models with Artificial Intelligence
## Tutorial at HAICON 2025 prologue day
![](data/banner.png)

### Organizers
[Alina Bazarova](https://www.fz-juelich.de/profile/bazarova_al), [Jose Robledo](https://www.fz-juelich.de/profile/robledo_j)

### Information

- **Date**: June 1st, 2025, 10:30-14:00
- **Place**: Karlsruhe, Germany (in-person event)
- [Link to Unconference page](https://haicon.cc/prologue-unconference/), and to [full description](https://github.com/Helmholtz-AI-Energy/HAICON25-Prologue-Day/issues/8).

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

**10.30 – 10.50** Introduction, tutorial overview, onboarding to HPC system

Teaching content

   - Overview of the tutorial
   - Conveying the learning objectives
   - Onboarding participants to JUWELS Booster HPC System

Learning Goals

   - Welcoming of the participants
   - Introduction to the system

**10.50 – 11.10** Lecture: Basic concepts of classical Bayesian inference

Teaching content

   - Key features of Bayesian Inference
   - Bayes rule
   - Concepts of posterior and prior distributions
   - Issues emerging when performing classical Bayesian Inference
   - Real life examples

Learning Goals

   - Getting insights into the theory behind Bayesian inference
   - Understanding the benefits of Bayesian inference through examples

**11.10 – 11.25** Hands-on: Warm-up example in a Jupyter notebook

Teaching content

   – A simple coin-flipping example implemented within Jupyter notebook

Learning Goals

   - Logging into the HPC systems and activating the necessary environment
   - Getting a feel of the prior and posterior distribution concepts

**11.25 - 11.40** Lecture: Basic concepts of Simulation Based Inference

Teaching content

   - A typical SBI pipeline
   - Key SBI methods and algorithmic discussion of those

Learning Goals

   - Understanding the benefits of SBI over the classical Bayesian inference
   - Understanding the difference between SBI methods

**11.40-12.00** Hands-on: Converting classical Bayesian example into an SBI one, Jupyter notebook

Teaching content

   - Using previously introduced example to write the first SBI pipeline
   - Compare different SBI methods on the same example

Learning goals

   - Set up the simplest one-liner interface of the SBI
   - Understand the difference in the inference and running times between the SBI methods

**12.00-13.00 Lunch break**

**13.00 - 13.15** Lecture: Deep Learning component and Sequential estimation

Teaching content

   - Estimation through normalizing flows: advantages and disadvantages
   - Estimation through a Neural Network classifier: parallels with MCMC
   - Concepts of Sequential Estimation

Learning Goals

   - Understand the machinery behind SBI
   - Consider potential benefits of the sequential estimation

**13.15 – 13.45** Hands-on: Flexible interface of the sbi package, Jupyter notebook

Teaching content

   - Utilise previously used data example to illustrate flexible interface of the SBI package
   - Customise neural network within the SBI machinery
   - Perform sequential inference on the same data example

Learning Goals

   - Work with SBI to a higher level of granularity
   - See the difference between amortized and sequential SBI inference

**13.45 – 14.30** Hands-on: Parallelization and distributing SBI over multiple nodes

Teaching content

   - Parallelise the simulations when using one node only
   - Distribute the simulations over multiple nodes by means of Ray backend
   - Use appropriate SLURM script for the corresponding batch job submission

Learning goals

   - Scale up the simulations in order to reduce the running time

### Useful links

- [sbi package GitHub repository](https://github.com/sbi-dev/sbi/tree/main/tutorials)

- [sbi workshop GitHub repository](https://github.com/mlcolab/sbi-workshop/tree/main/slides)

- [Probabilistic programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)


### :books: References

1. The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 - [Link to paper :newspaper:](https://doi.org/10.1073/pnas.191278911)

2. Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016 - [Link to paper :newspaper:](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)

3. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray - [Link to paper :newspaper:](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)

4. Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning - [Link to paper :newspaper:](http://proceedings.mlr.press/v119/hermans20a.html)






