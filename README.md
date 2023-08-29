# SecureLoop-MICRO2023Artifact

Artifacts of "SecureLoop: Design Space Exploration of Secure DNN Accelerators" (MICRO 2023)

This repository includes all source codes of SecureLoop and a Jupyter notebook to replicate the results in Figure 11. 

## Setup  

### Using the pre-built docker image

First, get the docker app (https://docs.docker.com/get-docker/). After cloning this repository, copy `docker-compose.yaml.template` to `docker-compose.yaml`:

```
cp docker-compose.yaml.template docker-compose.yaml
``` 

The docker image is hosted at dockerhub [timeloopaccelergy/timeloop-accelergy-pytorch:secureloop-amd64](https://hub.docker.com/layers/timeloopaccelergy/timeloop-accelergy-pytorch/secureloop-amd64/images/sha256-e1b04698aa53dde1eaf6a4e936f247e5b9aaf7dccb4bca37778eb2df8b322159?context=explore). Then, start the docker with:

```
docker-compose up
```

It launches the Jupyterlab hosted at `localhost:8888`. Note that depending on the docker environment, the IP address can be different from `localhost`.


### Run accelergyTables 

For the first time running this repo, open a terminal from Jupyterlab, and run:

```
accelergyTables -r effective_dram_tables/
```
This is only required for the first time, and does not need to be repeated. This step ensures that Accelergy recognizes the custom energy values for off-chip traffic including the cryptographic operations. 


### (Optional) Build docker images from the source

Instead of pulling the docker images from the dockerhub, you can build docker images from the sources directly. First, clone [the micro23-artifact-secureloop branch of the accelergy-timeloop-infrastructure repository](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/tree/micro23-artifact-secureloop), and build a docker image. Please read [SecureLoop-specific instructions](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/blob/micro23-artifact-secureloop/README-secureloop.md) before building. This image supports Timeloop with top-k search support and accelergy. 

Next, clone [the micro23-artifact-secureloop branch of the timeloop-accelergy-pytorch repository](https://github.com/Accelergy-Project/timeloop-accelergy-pytorch/tree/micro23-artifact-secureloop), and build a docker image. This image builds upon the `accelergy-timeloop-infrastructure:secureloop` docker image from the previous step, and supports PyTorch and Jupyterlab. 

## Reproducing Figure 11

We provide two methods to reproduce Figure 11. First, we provide a Jupyter notebook detailing all steps required to reproduce Figure 11. Second, we provide a script that runs all experiments necessary for Figure 11.

### Option 1: Run Jupyter Notebook

[`run_all.ipynb`](./workspace/run_all.ipynb) includes the step-by-step instructions to reproduce the results in Figure 11. Key steps are:

1. Define the DNN accelerator / cryptographic engine configurations
2. Define the DNN workload in PyTorch's `torch.nn.Module`
3. Run `timeloop-topk` to generate loopnest schedules for each layer of the workload DNN
4. Run `AuthBlockAssignment` to either calculate the overall latency/energy for the secure accelerator, or identify the optimal AuthBlock assignment strategy 
5. Run simulated annealing for joint search of multiple dependent layers in a DNN
6. Plot the graph with `matplotlib` (comparison of different scheduling algorithms)

The DNN accelerator / cryptographic engine setup is defined to match the one we used for Figure 11. However, for the design space exploration experiments (Figure 13, 14, 15, 16), the architecture configuration can be modified and the same notebook can be used. 

### Option 2: Script

We also provide a script that runs all experiments for Figure 11. Open a terminal from the JupyterLab, and run

```bash
bash scripts/fig11.sh
```

This script will first create an architecture description used in Figure 11, and the design will be saved in `designs/eyeriss_like/ver0`. Then, the script runs different scheduling algorithms presented in Figure 11, and dumps the result. After this script finishes, open [`plot_figures.ipynb`](./workspace/plot_figures.ipynb) and run all cells to plot and save the figures. 

### Notes on reproducing the results

SecureLoop's scheduling algorithm involves random processes (i.e., random pruning for timeloop, and simulated annealing). Thus, the result might not exactly match - however, the result should be within a close range to the values reported in the paper. For example, for the MobilenetV2 workload, using *Crypt-Opt-Cross* search algorithm results in the normalized latency of 9.83 in the paper. From multiple runs, we observe that this value can vary from 9.70 to 9.90. 

## Modifying Configurations

You can also run scheduling algorithms and obtain the performance metrics for different architecture configurations. 

### Generating architecture configurations

`workspace/designs/eyeriss_like/template` provides a template architecture for the `eyeriss_like` design we used in the paper. Based on this template, different configuration can be generated by varying the number of PEs, on-chip SRAM size, and cryptographic engines. Please refer to [`workspace/generate_arch.py`](./workspace/generate_arch.py) for details. 

As an example, a design with the same accelerator configuration but with a "parallel" AES-GCM cryptographic engine can be generated by

```bash
python3 generate_arch.py --crypt_engine_type parallel
```

Running the python script will generate a new folder in `workspace/designs/eyeriss_like`. If the configuration matches with the already existing one, it will not generate a new folder.

If you want to use a different template (e.g., architectures with different dataflow), you can define the template in `workspace/designs/{arch_name}/template`, and generate configurations based on that template as well.  

### Run scheduling

Run scheduling algorithms for secure accelerator configurations using [`workspace/scheduler.py`](./workspace/scheduler.py):

```bash
python3 scheduler.py --arch [architecture config path] --workload [workload name] --scheduler [scheduler name] --topk [integer k for top-k search] 
```

For example, running *crypt-opt-cross* scheduling for the architecture configuration in `workspace/designs/eyeriss_like/ver0` with the `mobilenet_v2` workload will be:

```bash
python3 scheduler.py --arch designs/eyeriss_like/ver0 --workload mobilenet_v2 --scheduler crypt-opt-cross
```

For details of options, please refer to `workspace/scheduler.py`. 


### Check results

Running `scheduler.py` with *crypt-opt-cross* algorithm dumps the result into `[architecture config path]/joint_topk/crypt-opt-cross_cost.yaml` and `[architecture config path]/joint_topk/search_joint.csv`. For *crypt-tile-single* and *crypt-opt-single*, the results are dumped into `[architecture config path]/evaluation` folder. Check the csv file for the summary of schedules and stats for each layer and inter-layer dependency.  

## Code Organization

Other notebooks explain more details for each step in our scheduler:

+ `run_loopnest_scheduling.ipynb`: Running `timeloop-topk` to generate loopnest schedules
+ `run_authblock_assignment.ipynb`: Details of AuthBlock assignment, involving the additional off-chip traffic and cryptographic action counting. All functions explained in this notebook is wrapped up in `authblock_assignment.py` 
+ `run_simulated_annealing.ipynb`: Running simulated annealing for joint search of multiple layers

`plot_figures.ipynb` runs utility functions for plotting Figure 11 (see above). 

Source codes include:

+ Python scripts:
   + `generate_arch.py`: generate architecture configurations based on the template
   + `scheduler.py`: runs scheduling algorithms for a given architecture configuration, workload, and the scheduler algorithm
+ Source codes in python supporting the AuthBlock assignment and other utils:
    + `authblock_assignment.py`: AuthBlock assignment top-level function
    + `authentication_block_assignment_utils.py`: util functions such as extracting tile information from the timeloop stats, identify the overlaps/halos/interlayer dependency between tiles/layers etc.
    + `pytorch_layer_dependency_utils.py`: util functions for analyzing the PyTorch model's back-propagation graph to determine which layers are consecutive and dependent
    + `summary_writer.py`: dumping the AuthBlock assignment result to csv files
    + `testbench_utils.py`: functions for calculating the additional off-chip traffic and cryptographic action counts; used in `authblock_assignment.py`
    + `tile_analysis_utils.py`: util functions for basic tile analysis (parsing the timeloop xml stats file, getting tile sizes, etc.)
    + `utils.py`: util function for generating architecture configurations, used in the Jupyter notebooks
+ `designs/`: DNN architecture descriptions for Timeloop. Scheduling results are also dumped to this folder. 
+ `effective_dram_tables/`: tables for Accelergy table plug-ins. Equivalent energy model considering cryptographic engines.
+ `pytorch2timeloop/`: local version of PyTorch workloads to Timeloop workloads converter
+ `workloads/`: the converted workloads are stored here  

