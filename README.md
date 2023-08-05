# SecureLoop-MICRO2023Artifact

Artifacts of "SecureLoop: Design Space Exploration of Secure DNN Accelerators" (MICRO 2023)

This repository includes all source codes of SecureLoop and the Jupyter notebook to replicate the results in Figure 11. 

## Setup  

All codes and Jupyter notebooks are located in `workspace` folder. First, start the docker with 


```
docker-compose up
```

It launches the Jupyter notebook hosted at `localhost:8888`. Note that depending on the docker environment, the IP address can be different from `localhost`. 

## Reproducing Figure 11

`run_all.ipynb` includes the step-by-step instructions to reproduce the results in Figure 11. Key steps are:

1. Define the DNN accelerator / cryptographic engine configurations
2. Define the DNN workload in PyTorch's `torch.nn.Module`
3. Run `timeloop-topk` to generate loopnest schedules for each layer of the workload DNN
4. Run `AuthBlockAssignment` to either calculate the overall latency/energy for the secure accelerator, or identify the optimal AuthBlock assignment strategy 
5. Run simulated annealing for joint search of multiple dependent layers in a DNN
6. Plot the graph with `matplotlib` (comparison of different scheduling algorithms)

The DNN accelerator / cryptographic engine setup is defined to match the one we used for Figure 11. However, for the design space exploration experiments (Figure 13, 14, 15, 16), the architecture configuration can be modified and the same notebook can be used. 


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

