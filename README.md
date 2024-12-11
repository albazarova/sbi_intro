# Introduction to Simulation Based Inference: enhancing synthetic models with Artificial Intelligence
### :high_brightness: Welcome to the tutorial on Simulation Based Inference with HPCs

### Organizers
[Alina Bazarova](https://www.fz-juelich.de/profile/bazarova_al), [Jose Robledo](https://www.fz-juelich.de/profile/robledo_j), and [Stefan Kesselheim](https://www.google.com/search?client=ubuntu-sn&channel=fs&q=stefan+Kesselheim)

**Description** This is a Simulation Based Inference (SBI) tutorial which is based on the python package [sbi](https://github.com/sbi-dev/sbi) extended with a patch to be distributed accross multiple High Performance Computing (HPC) nodes. It uses a number of toy data/model examples, as well as simplified examples of real-life problems taken from the authors'research. 

You may find the tutorial notebooks in the [notebooks](./notebooks) folder. This course is intended for half a day (4 hours), where we introduce the SBI methodology and how to implement it by means of the [`sbi` package](https://github.com/sbi-dev/sbi) in HPC clusters. We provide an example multi node execution in the  [multi_node_ray](./multi_node_ray/) folder.

The material of this tutorial partly uses the examples from the repositories listed below:

- [sbi package GitHub repository](https://github.com/sbi-dev/sbi/tree/main/tutorials)

- [sbi workshop GitHub repository](https://github.com/mlcolab/sbi-workshop/tree/main/slides)

- [Probabilistic programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)


### :books: Useful references:

1. The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 - [Link to paper :newspaper:](https://doi.org/10.1073/pnas.191278911)

2. Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016 - [Link to paper :newspaper:](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)

3. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray - [Link to paper :newspaper:](http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf)

4. Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning - [Link to paper :newspaper:](http://proceedings.mlr.press/v119/hermans20a.html)

### :rocket: Working with [ray cluster](https://docs.ray.io/en/latest/index.html)

An example of how to submit jobs on a JSC cluster (JSC) using with ray backend for distributed simulation and training can be found in the `run_ray.sbatch` script. There is a virtual environment prepared to run on JSC clusters in the `multi_node_ray` folder. You will need a judoor account for this.

If it is the first time you are using this repo on the cluster, head to the `multsbi_env` folder and run the `setup.sh` shell script to create the required python environment.

```bash
cd sbi_env
source setup.sh
cd ../
```

Afterwards, use `source ray_cluster/activate.sh` to activate the corresponding environment. Then head to the `multi_node_ray` folder and use the batch script `run_ray.sbatch` in order to submit a job to run `pyross_example_script.py`. Remember to modify the header of the sbatch file accordingly to your account, partition, and resources.

```bash
source sbi_env/activate.sh
cd multi_node_ray
sbatch run_ray.sbatch
```








