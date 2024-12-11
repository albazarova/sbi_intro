# :rocket: Working with [ray cluster](https://docs.ray.io/en/latest/index.html)

An example of how to submit jobs on a JSC cluster (JSC) using with ray backend for distributed simulation and training can be found in the `run_ray.sbatch` script. There is a virtual environment prepared to run on JSC clusters in the `multi_node_ray` folder. You will need a judoor account for this.

If it is the first time you are using this repo on the cluster, head to the `multsbi_env` folder and run the `setup.sh` shell script to create the required python environment. If you are standing in this folder, then run

```bash
cd ../sbi_env
source setup.sh
cd ../
```

Afterwards, use `source ray_cluster/activate.sh` to activate the corresponding environment. Then head to the `multi_node_ray` folder and use the batch script `run_ray.sbatch` in order to submit a job to run `pyross_example_script.py`. Remember to modify the header of the sbatch file accordingly to your account, partition, and resources.

```bash
source sbi_env/activate.sh
cd multi_node_ray
sbatch run_ray.sbatch
```