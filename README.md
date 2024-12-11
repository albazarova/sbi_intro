# Introduction to SBI (and Scaling!)
### Tutorial on Simulation Based Inference


Material of this tutorial partly uses the examples from the repositories listed below:

https://github.com/sbi-dev/sbi/tree/main/tutorials

https://github.com/mlcolab/sbi-workshop/tree/main/slides

https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers


### Useful references:

The frontier of simulation-based inference, Kyle Cranmer, Johann Brehmer, and Gilles Louppe, PNAS 117 (48) 30055-30062 https://doi.org/10.1073/pnas.191278911

Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation, George Papamakarios, Iain Murray, NeurIPS 2016, https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf

Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows, George Papamakarios, David C. Sterratt, Iain Murray, http://proceedings.mlr.press/v89/papamakarios19a/papamakarios19a.pdf

Likelihood-free MCMC with Amortized Approximate Likelihood Ratios, Joeri Hermans, Volodimir Begy, Gilles Louppe Proceedings of the 37th International Conference on Machine Learning http://proceedings.mlr.press/v119/hermans20a.html

### Working with ray cluster

An example of how to submit jobs on a JSC cluster (JSC) using with ray backend for distributed simulation and training can be found in the `run_ray.sbatch` script. There is a virtual environment prepared to run on JSC clusters in the `ray_cluster` folder. You will need a judoor account for this.

If it is the first time you are using this repo on the cluster, head to the `ray_cluster` folder (`$cd ray_cluster`) and run `bash setup.sh` to create the required python environment.

Afterwards, use `source ray_cluster/activate.sh` to activate the corresponding environment. Then use the batch script `run_ray.sbatch` in order to submit a job to run `pyross_example_script.py`








