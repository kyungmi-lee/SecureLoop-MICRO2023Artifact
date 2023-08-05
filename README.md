# SecureLoop-MICRO2023Artifact

Artifacts of "SecureLoop: Design Space Exploration of Secure DNN Accelerators" (MICRO 2023)

This repository includes all source codes of SecureLoop and a Jupyter notebook to replicate the results in Figure 11. 

## Setup  

First, get the docker app (https://docs.docker.com/get-docker/). After cloning this repository, copy `docker-compose.yaml.template` to `docker-compose.yaml`:

```
cp docker-compose.yaml.template docker-compose.yaml
``` 

The docker image is hosted at dockerhub [timeloopaccelergy/timeloop-accelergy-pytorch:secureloop-amd64](https://hub.docker.com/layers/timeloopaccelergy/timeloop-accelergy-pytorch/secureloop-amd64/images/sha256-e1b04698aa53dde1eaf6a4e936f247e5b9aaf7dccb4bca37778eb2df8b322159?context=explore). Then, start the docker with:

```
docker-compose up
```

It launches the Jupyterlab hosted at `localhost:8888`. Note that depending on the docker environment, the IP address can be different from `localhost`.

## Reproducing Figure 11

`run_all.ipynb` includes the step-by-step instructions to reproduce the results in Figure 11. Key steps are:

1. Define the DNN accelerator / cryptographic engine configurations
2. Define the DNN workload in PyTorch's `torch.nn.Module`
3. Run `timeloop-topk` to generate loopnest schedules for each layer of the workload DNN
4. Run `AuthBlockAssignment` to either calculate the overall latency/energy for the secure accelerator, or identify the optimal AuthBlock assignment strategy 
5. Run simulated annealing for joint search of multiple dependent layers in a DNN
6. Plot the graph with `matplotlib` (comparison of different scheduling algorithms)

The DNN accelerator / cryptographic engine setup is defined to match the one we used for Figure 11. However, for the design space exploration experiments (Figure 13, 14, 15, 16), the architecture configuration can be modified and the same notebook can be used. 

### Notes on reproducing the results

SecureLoop's scheduling algorithm involves random processes (i.e., random pruning for timeloop, and simulated annealing). Thus, the result might not exactly match - however, the result should be within a close range to the values reported in the paper. For example, for the MobilenetV2 workload, using *Crypt-Opt-Cross* search algorithm results in the normalized latency of 9.83 in the paper. From multiple runs, we observe that this value can vary from 9.81 to 9.90. 

## Code Organization

Other notebooks explain more details for each step in our scheduler:

+ `run_loopnest_scheduling.ipynb`: Running `timeloop-topk` to generate loopnest schedules
+ `run_authblock_assignment.ipynb`: Details of AuthBlock assignment, involving the additional off-chip traffic and cryptographic action counting. All functions explained in this notebook is wrapped up in `authblock_assignment.py` 
+ `run_simulated_annealing.ipynb`: Running simulated annealing for joint search of multiple layers

Source codes include:

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

