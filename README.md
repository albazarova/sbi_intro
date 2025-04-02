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


### Useful links

- [sbi package GitHub repository](https://github.com/sbi-dev/sbi/tree/main/tutorials)

- [sbi workshop GitHub repository](https://github.com/mlcolab/sbi-workshop/tree/main/slides)

- [Probabilistic programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)


### :books: References

1. The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 - [Link to paper :newspaper:](https://doi.org/10.1073/pnas.191278911)

2. Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016 - [Link to paper :newspaper:](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)

3. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray - [Link to paper :newspaper:](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)

4. Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning - [Link to paper :newspaper:](http://proceedings.mlr.press/v119/hermans20a.html)






