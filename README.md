# :high_brightness: Simulation-Based Inference for Computational Biology: Integrating AI, Bayesian Modeling, and HPC
## Tutorial IP2 - ISCB-AFRICA ASBCB Conference on Bioinformatics
![](https://www.iscb.org/images/stories/africa2025/banner.ConferenceBanner.Africa.2025.png)

### Organizers
[Alina Bazarova](https://www.fz-juelich.de/profile/bazarova_al), [Jose Robledo](https://www.fz-juelich.de/profile/robledo_j), and [Stefan Kesselheim](https://www.google.com/search?client=ubuntu-sn&channel=fs&q=stefan+Kesselheim)

### Information

- **Date**: April 17, 2025, 09:00-13:00
- **Place**: Capetown, South-Africa (in-person event)
- [Link to tutorial](https://www.iscb.org/africa2025/programme-agenda/tutorials#ip2) on the ISCB-AFRICA ASBCB Conference on Bioinformatics Homepage.

### Description
This tutorial introduces Simulation-Based Inference (SBI), a framework combining Bayesian modeling, AI techniques, and high-performance computing (HPC) to address key challenges in computational biology, such as performing reliable inference with limited data by using AI-based approximate Bayesian computation. Moreover, it tackles the problem of intractable likelihood functions, thereby allowing to utilize Bayesian inference for biological systems with multiple sources of stochasticity. The tutorial also demonstrates how to leverage HPC environments to drastically reduce inference runtimes, making it highly relevant for large-scale biological problems. This tutorial bridges theoretical foundations with hands-on applications in computational biology.

Participants will learn to implement SBI frameworks using diverse biological models, such as molecular dynamics simulations, agent-based tumor growth models, count data modeling, and Lotka-Volterra systems. Practical exercises in Jupyter notebooks guide attendees through SBI workflows, from simple coin-flipping examples to more complex biological simulations, ensuring accessibility for participants with varied backgrounds. The tutorial’s inclusion of cutting-edge methods like Sequential Neural Posterior Estimation and its emphasis on parallelization and HPC scalability align closely with the scientific community's focus on innovation in computational biology. A previous iteration of the tutorial at the Helmholtz AI Conference 2024 received excellent reviews and led to interdisciplinary discussions, highlighting its broad applicability and impact. For this conference, the content has been further refined with additional examples relevant to the community, ensuring it meets the needs of bioinformatics researchers.

You may find the tutorial notebooks in the [notebooks](./notebooks) folder.

### Learning Objectives

- Understand the Principles of Simulation-Based Inference (SBI): learn the theoretical foundations of SBI, including its relationship with Bayesian inference and its advantages in handling complex biological systems.
- Explore SBI Methods (SNPE, SNLE, and SNRE): gain an understanding of Sequential Neural Posterior Estimation (SNPE), Sequential Neural Likelihood Estimation (SNLE), and Sequential Neural Ratio Estimation (SNRE) and their applications in computational biology.
- Learn how to design and implement SBI frameworks for representative biological scenarios, such as molecular dynamics, cell growth, count data modeling, and Lotka-Volterra systems.
- Leverage HPC for SBI Workflows: understand how to use high-performance computing (HPC) environments to scale SBI workflows and efficiently distribute computational workloads.

### Course schedule

Course schedule:
 
**9.00 – 9.30** Introduction, tutorial overview, onboarding to HPC system
            
Teaching content
- Overview of the tutorial
- Conveying the learning objectives
- Onboarding participants to JUSUF HPC System

Learning Goals
- Welcoming of the participants
- Introduction to the system
 
**9.30 – 10.00** Lecture: Basic concepts of classical Bayesian inference
            
Teaching content
- Key features of Bayesian Inference
- Bayes rule
- Concepts of posterior and prior distributions
- Issues emerging when performing classical Bayesian Inference
- Real life examples
- A typical SBI pipeline
- Key SBI methods and algorithmic discussion of those          

Learning Goals
- Getting insights into the theory behind Bayesian inference
- Understanding the benefits of Bayesian inference through examples
- Understanding the benefits of SBI over the classical Bayesian inference
- Understanding the difference between SBI methods
                                                                        
                        
**10.00-10.15** Hands-on: Converting classical Bayesian example into an SBI one, Jupyter notebook
 
Teaching content
- Using previously introduced example to write the first SBI pipeline
- Compare different SBI methods on the same example
  
Learning goals
- Logging into the HPC systems and activating the necessary environment
- Getting a feel of the prior and posterior distribution concepts 
- Set up the simplest one-liner interface of the SBI
- Understand the difference in the inference and running times between the SBI methods
 
**10.15 - 10.45** Hands-on: Data example. MCMC vs SBI, Jupyter notebook
 
Teaching content
- Using a data example to run classical MCMC algorithm to get parameter     estimates
- Put the example into the SBI framework

Learning Goals
- Learn to adapt a more complex example to SBI framework
- Evaluate the differences between SBI and classical Bayesian inference
 
**10.45 - 11.00** Coffee-break
 
**11.00 - 11.15** Lecture: Deep Learning component and Sequential estimation
 
Teaching content
- Estimation through normalizing flows: advantages and disadvantages
- Estimation through a Neural Network classifier: parallels with MCMC
- Concepts of Sequential Estimation
  
Learning Goals
- Understand the machinery behind SBI
- Consider potential benefits of the sequential estimation
            
**11.15 – 11.45** Hands-on: Flexible interface of the sbi package, Jupyter notebook 
            
Teaching content
- Utilise previously used data example to illustrate flexible interface of the SBI package
- Customise neural network within the SBI machinery
- Perform sequential inference on the same data example

Learning Goals
- Work with SBI to a higher level of granularity
- See the difference between amortized and sequential SBI inference
 
**11.45 – 12.20** Hands-on: Parallelization and distributing SBI over multiple nodes
            
Teaching content
- Parallelise the simulations when using one node only
- Distribute the simulations over multiple nodes by means of Ray backend
- Use appropriate SLURM script for the corresponding batch job submission

Learning goals
- Scale up the simulations in order to reduce the running time
 
**12.20 – 13.00** Hands-on: Constructing summary statistics
            
Teaching content
- Introduction to a 1D molecular dynamics simulator example
- Extract summary statistic from the model
- Compare the outputs given different parameters of the summary statistic
- Evaluate the results
- Introduction to an agent based simulator for tumor cell growth (optional)

Learning goals
- Learn summary statistics from images by means of CNNs (optional)
- Craft summary statistics based on domain knowledge
- Consider neural networks to learn new summary statistics
- Familiarize with different types of summary statistics and their impact on the inference
- Familiarize with the flexible SBI interface
 

### Useful links

- [sbi package GitHub repository](https://github.com/sbi-dev/sbi/tree/main/tutorials)

- [sbi workshop GitHub repository](https://github.com/mlcolab/sbi-workshop/tree/main/slides)

- [Probabilistic programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)


### :books: References

1. The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 - [Link to paper :newspaper:](https://doi.org/10.1073/pnas.191278911)

2. Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016 - [Link to paper :newspaper:](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)

3. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray - [Link to paper :newspaper:](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)

4. Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning - [Link to paper :newspaper:](http://proceedings.mlr.press/v119/hermans20a.html)






