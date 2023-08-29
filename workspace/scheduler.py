import os
import yaml
import shutil
from pathlib import Path
import argparse
import random
import time
import csv
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo

import pytorch2timeloop as pytorch2timeloop
from pytorch_layer_dependency_utils import BackpropGraph
from authblock_assignment import AuthBlockAssignment
from authblock_assignment import PartialUpdateAuthBlockAssignment
from utils import xml2mapping 

def extract_layer_info(net, input_size, base_dir, top_dir, sub_dir):
    n_layers = 0
    layer_dict = {}
    layer_duplicate_info = {}
    unique_layers = []
    for module in net.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            n_layers += 1
            if n_layers not in layer_dict.keys():
                workload_path = os.path.join(base_dir, top_dir, sub_dir, '{}_layer{}.yaml'.format(sub_dir, n_layers))
                with open(workload_path, 'r') as f:
                    workload_info = yaml.safe_load(f)
                layer_dict[n_layers] = workload_info
            
            # identify the earliest duplicate layer
            for key in range(1, n_layers):
                if layer_dict[key] == layer_dict[n_layers]:
                    layer_duplicate_info[n_layers] = key
                    break
            if n_layers not in layer_duplicate_info:
                unique_layers.append(n_layers)

    workload_path_1 = os.path.join(base_dir, top_dir, sub_dir, 'layer_info_interlayer.yaml')
    workload_path_2 = os.path.join(base_dir, top_dir, sub_dir, 'layer_info_ignore_interlayer.yaml') # we also need this for baseline

    try:
        with open(workload_path_1, 'r') as f:
            layer_info = yaml.safe_load(f)
        with open(workload_path_2, 'r') as f:
            layer_info_ignore_interlayer = yaml.safe_load(f)
            
    except:
        graph = BackpropGraph(net, [1, input_size[0], input_size[1], input_size[2]])
        consecutive_dict, dependent_dict = graph.get_dependency_info()
    
        # construct layer_info
        layer_info = {}
        layer_info_ignore_interlayer = {}
        for layer_idx in range(1, n_layers + 1):
            info = {}
            if layer_idx in unique_layers:
                info['layer_id_for_timeloop'] = layer_idx
            else:
                info['layer_id_for_timeloop'] = layer_duplicate_info[layer_idx]
            info['prev_layer'] = []
            info['next_layer'] = []
            info['dependent_prev_layer'] = []
            info['dependent_next_layer'] = []
            layer_info[layer_idx] = info
            layer_info_ignore_interlayer[layer_idx] = info
    
        for layer_idx in range(1, n_layers + 1):
            consecutive = consecutive_dict[layer_idx]
            dependent = dependent_dict[layer_idx]
            layer_info[layer_idx]['next_layer'].extend(consecutive)
            layer_info_ignore_interlayer[layer_idx]['next_layer'].extend(consecutive)
            for i in consecutive:
                layer_info[i]['prev_layer'].append(layer_idx)
                layer_info_ignore_interlayer[i]['prev_layer'].append(layer_idx)
            if len(dependent) > 0 and not ignore_interlayer:
                layer_info[layer_idx]['dependent_next_layer'].extend(dependent)   
                for i in dependent:
                    layer_info[i]['dependent_prev_layer'].append(layer_idx)

    return n_layers, unique_layers, layer_info, layer_info_ignore_interlayer

def run_timeloop(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, topk, base=False):
    def get_cmd(workload_info, layer_id, base_dir, timeloop_dir, sub_dir, top_dir, base):
        if base:
            cwd = f"{base_dir/timeloop_dir/'baseline_scheduling'/sub_dir/f'layer{layer_id}'}"
        else:
            cwd = f"{base_dir/timeloop_dir/'scheduling'/sub_dir/f'layer{layer_id}'}"
        if 'M' in workload_info['problem']['instance']:
            constraint_pth = base_dir/timeloop_dir/'constraints/*.yaml'
        else:
            # depthwise
            constraint_pth = base_dir/timeloop_dir/'constraints_dw/*.yaml'

        if base:
            timeloopcmd = f"timeloop-mapper-topk " \
                          f"{base_dir/timeloop_dir/'arch/baseline.yaml'} " \
                          f"{base_dir/timeloop_dir/'arch/components/*.yaml'} " \
                          f"{base_dir/timeloop_dir/'mapper/mapper.yaml'} " \
                          f"{constraint_pth} " \
                          f"{base_dir/top_dir/sub_dir/sub_dir}_layer{layer_id}.yaml "
        else:
            timeloopcmd = f"timeloop-mapper-topk " \
                          f"{base_dir/timeloop_dir/'arch/effective.yaml'} " \
                          f"{base_dir/timeloop_dir/'arch/components/*.yaml'} " \
                          f"{base_dir/timeloop_dir/'mapper/mapper.yaml'} " \
                          f"{constraint_pth} " \
                          f"{base_dir/top_dir/sub_dir/sub_dir}_layer{layer_id}.yaml "
            
        return [cwd, timeloopcmd]

    cwd_list = []
    cmd_list = []

    for layer_id in unique_layers:
        workload_path = os.path.join(base_dir, top_dir, sub_dir, '{}_layer{}.yaml'.format(sub_dir, layer_id))
        with open(workload_path, 'r') as f:
            workload_info = yaml.safe_load(f)
        [cwd, cmd] = get_cmd(workload_info, layer_id, base_dir, timeloop_dir, sub_dir, top_dir, base)
        cwd_list.append(cwd)
        cmd_list.append(cmd)
        
    if not os.path.exists(os.path.join(base_dir, timeloop_dir, 'scheduling' if not base else 'baseline_scheduling', sub_dir)):
        os.mkdir(os.path.join(base_dir, timeloop_dir, 'scheduling' if not base else 'baseline_scheduling', sub_dir))

    for cwd, cmd in zip(cwd_list, cmd_list):
        print("Executing cmd: {}".format(cmd))
        try:
            os.chdir(cwd)
        except:
            os.mkdir(cwd)
            os.chdir(cwd)
        os.system(cmd)
    os.chdir(base_dir)

    def convert_to_mapping(base_dir, timeloop_dir, top_dir, sub_dir, layer_idx, topk_idx, base=False):
        if base:
            xml_file = os.path.join(base_dir, timeloop_dir, 'baseline_scheduling', sub_dir, "layer{}".format(layer_idx), \
                                    "timeloop-mapper-topk{}.map+stats.xml".format(topk_idx))
        else:
            xml_file = os.path.join(base_dir, timeloop_dir, 'scheduling', sub_dir, "layer{}".format(layer_idx), \
                                    "timeloop-mapper-topk{}.map+stats.xml".format(topk_idx))
        workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_idx))

        with open(workload_file, 'r') as f:
            workload_info = yaml.safe_load(f)
        if 'M' in workload_info['problem']['instance']:
            dw = False
        else:
            dw = True
        arch_constraint_file = os.path.join(base_dir, timeloop_dir, 'constraints_dw' if dw else 'constraints' , \
                                            'eyeriss_like_arch_constraints.yaml')
        mapping = xml2mapping(xml_file, workload_file, arch_constraint_file, dw)
        with open(os.path.join(base_dir, timeloop_dir, 'scheduling' if not base else 'baseline_scheduling', sub_dir, "layer{}".format(layer_idx), \
                               "mapping{}.yaml".format(topk_idx)), 'w') as f:
            _ = yaml.dump({'mapping': mapping}, f)
            
    for layer_idx in unique_layers:
        for k in range(1, topk + 1):
            convert_to_mapping(base_dir, timeloop_dir, top_dir, sub_dir, layer_idx, k, base)

def run_timeloop_model(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, base=False):
    def get_cmd_model(workload_info, layer_id, base_dir, timeloop_dir, sub_dir, top_dir, base, k):
        if base:
            cwd = f"{base_dir/timeloop_dir/'baseline_evaluation'/sub_dir/f'layer{layer_id}'}"
        else:
            cwd = f"{base_dir/timeloop_dir/'evaluation'/sub_dir/f'layer{layer_id}'}"
        if 'M' in workload_info['problem']['instance']:
            constraint_pth = base_dir/timeloop_dir/'constraints/*.yaml'
        else:
            # depthwise
            constraint_pth = base_dir/timeloop_dir/'constraints_dw/*.yaml'

        if base:
            timeloopcmd = f"timeloop-model " \
                          f"{base_dir/timeloop_dir/'arch/baseline.yaml'} " \
                          f"{base_dir/timeloop_dir/'arch/components/*.yaml'} " \
                          f"{base_dir/timeloop_dir/'baseline_scheduling'/sub_dir/f'layer{layer_id}'/f'mapping{k}.yaml'} " \
                          f"{base_dir/top_dir/sub_dir/sub_dir}_layer{layer_id}.yaml "
        else:
            timeloopcmd = f"timeloop-model " \
                          f"{base_dir/timeloop_dir/'arch/baseline.yaml'} " \
                          f"{base_dir/timeloop_dir/'arch/components/*.yaml'} " \
                          f"{base_dir/timeloop_dir/'scheduling'/sub_dir/f'layer{layer_id}'/f'mapping{k}.yaml'} " \
                          f"{base_dir/top_dir/sub_dir/sub_dir}_layer{layer_id}.yaml "
        return [cwd, timeloopcmd]
        
    cwd_list = []
    cmd_list = []
    for layer_id in unique_layers:
        workload_path = os.path.join(base_dir, top_dir, sub_dir, '{}_layer{}.yaml'.format(sub_dir, layer_id))
        with open(workload_path, 'r') as f:
            workload_info = yaml.safe_load(f)
        [cwd, cmd] = get_cmd_model(workload_info, layer_id, base_dir, timeloop_dir, sub_dir, top_dir, base, 1)
        cwd_list.append(cwd)
        cmd_list.append(cmd)
        
    if not os.path.exists(os.path.join(base_dir, timeloop_dir, 'evaluation' if not base else 'baseline_evaluation', sub_dir)):
        os.mkdir(os.path.join(base_dir, timeloop_dir, 'evaluation' if not base else 'baseline_evaluation', sub_dir))
    for cwd, cmd in zip(cwd_list, cmd_list):
        print("Executing cmd: {}".format(cmd))
        try:
            os.chdir(cwd)
        except:
            os.mkdir(cwd)
            os.chdir(cwd)
        os.system(cmd)
    os.chdir(base_dir)

def prepare_for_simulated_annealing(n_layers, layer_info, base_dir, timeloop_dir, top_dir, sub_dir, configuration_dict, topk):
    base_cost_dict, base_rehash_cost_dict, base_block_info_dict = AuthBlockAssignment(n_layers, layer_info, \
                                                                                      base_dir, timeloop_dir, top_dir, sub_dir, \
                                                                                      configuration_dict, \
                                                                                      mode="search", \
                                                                                      joint=False, return_cost_dict=True)
    
    baseline_energy = 0
    baseline_latency = 0
    baseline_add_mem_traffic = 0
    
    for key in base_cost_dict:
        baseline_energy += base_cost_dict[key]['total_energy'] / 10**6
        baseline_latency += base_cost_dict[key]['total_latency']
        baseline_add_mem_traffic += base_cost_dict[key]['add_memory_traffic']
    for key in base_rehash_cost_dict:
        baseline_energy += base_rehash_cost_dict[key]['total_energy'] / 10**6
        baseline_latency += base_rehash_cost_dict[key]['total_latency']
        baseline_add_mem_traffic += base_rehash_cost_dict[key]['add_memory_traffic']   
        
    for layer_idx in range(1, n_layers + 1):
        work_dir = os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'layer{}'.format(layer_idx))
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
            
        # """
        for k in range(1, topk + 1):
            if not os.path.exists(os.path.join(work_dir, 'eval{}'.format(k))):
                os.mkdir(os.path.join(work_dir, 'eval{}'.format(k)))
            layer_id_for_timeloop = layer_info[layer_idx]['layer_id_for_timeloop']
            cwd = f"{base_dir/timeloop_dir/'joint_topk'/sub_dir/f'layer{layer_idx}'/f'eval{k}'}"
    
            timeloopcmd = f"timeloop-model " \
                  f"{base_dir/timeloop_dir/'arch/baseline.yaml'} " \
                  f"{base_dir/timeloop_dir/'arch/components/*.yaml'} " \
                  f"{base_dir/timeloop_dir/'scheduling'/sub_dir/f'layer{layer_id_for_timeloop}'/f'mapping{k}.yaml'} " \
                  f"{base_dir/top_dir/sub_dir/sub_dir}_layer{layer_idx}.yaml "
            
            try:
                os.chdir(cwd)
            except:
                os.mkdir(cwd)
                os.chdir(cwd)
            os.system(timeloopcmd)
            os.chdir(base_dir)
        # """
    
        # copy mapping1's result into here
        shutil.copy(os.path.join(work_dir, 'eval1', 'timeloop-model.map+stats.xml'), work_dir)
        
    return base_cost_dict, base_rehash_cost_dict, base_block_info_dict, baseline_energy, baseline_latency, baseline_add_mem_traffic

def run_simulated_annealing(n_layers, layer_info, base_dir, timeloop_dir, top_dir, sub_dir, configuration_dict, topk, layers_exclude_from_search, \
                            model_name):

    base_cost_dict, base_rehash_cost_dict, base_block_info_dict, baseline_energy, baseline_latency, baseline_add_mem_traffic = \
    prepare_for_simulated_annealing(n_layers, layer_info, base_dir, timeloop_dir, top_dir, sub_dir, configuration_dict, topk)
    
    # TODO: add options to change these hyperparams
    initial_temp = 100
    final_temp = 0.1
    n_iters = 1000
    
    cooling_scheduler = 'linear'
    
    # TODO: this option should not be used for ResNet18 - bug with dependent layer partial update due to residuals
    use_partial_update = True
    if model_name == 'resnet18':
        use_partial_update = False

    csv_header = ['Iter', 'Temp', \
                  'Cost (J x cycles)', 'Total Latency (cycles)', 'Total Energy (uJ)', 'Additional Off-chip Traffic (bits)']
    logs = []
    
    solution_cost_dict = copy.deepcopy(base_cost_dict)
    solution_rehash_cost_dict = copy.deepcopy(base_rehash_cost_dict)
    solution_block_info_dict = copy.deepcopy(base_block_info_dict)
    
    current_cost_dict = copy.deepcopy(base_cost_dict)
    current_rehash_cost_dict = copy.deepcopy(base_rehash_cost_dict)
    current_block_info_dict = copy.deepcopy(base_block_info_dict)
    
    solution_state = [1] * n_layers
    current_state = [1] * n_layers
    best_state = [1] * n_layers
    
    i = 0
    cost_best = baseline_energy * baseline_latency
    
    layers_for_search = []
    for idx in range(1, n_layers + 1):
        if len(layer_info[idx]['dependent_next_layer']) > 0 or len(layer_info[idx]['dependent_prev_layer']) > 0:
            if idx not in layers_exclude_from_search:
                layers_for_search.append(idx)
                
    # start_time = time.time()
    while i < n_iters + 1:
        # temperature
        if cooling_scheduler == 'linear':
            current_temp = final_temp + (initial_temp - final_temp) / float(n_iters) * float(n_iters - i)
        elif cooling_scheduler == 'cosine':
            current_temp = final_temp + 0.5 * (initial_temp - final_temp) * (1 + math.cos(float(i) * math.pi / float(n_iters)))
        elif cooling_scheduler == 'quadratic':
            current_temp = final_temp + (initial_temp - final_temp) * (float(n_iters - i) / float(n_iters))**2
        
        layer2change = random.choice(layers_for_search)
        neighbor_loopnest = random.choice(list(range(1, topk + 1)))
        
        current_state[layer2change - 1] = neighbor_loopnest
        stats_file = os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, "layer{}".format(layer2change), \
                                  "eval{}".format(neighbor_loopnest), "timeloop-model.stats.txt")
        with open(stats_file, 'r') as f:
            lines = f.read().split('\n')[-200:]
            for line in lines:
                if line.startswith('Energy'):
                    energy = eval(line.split(': ')[1].split(' ')[0]) * float(10**6) # micro to pico
                    # print(energy)
                elif line.startswith('Cycles'):
                    cycle = eval(line.split(': ')[1])
        current_cost_dict[layer2change]['timeloop_energy'] = energy
        current_cost_dict[layer2change]['timeloop_cycle'] = cycle
        
        xml_file = os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, "layer{}".format(layer2change), \
                                "eval{}".format(neighbor_loopnest), "timeloop-model.map+stats.xml")
        shutil.copy(xml_file, os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'layer{}'.format(layer2change)))
        
        if use_partial_update:
            subset_layers = [layer2change]
            subset_layers.extend(layer_info[layer2change]['prev_layer'])
            subset_layers.extend(layer_info[layer2change]['next_layer'])
            
            current_cost_dict, current_rehash_cost_dict, current_block_info_dict = \
            PartialUpdateAuthBlockAssignment(n_layers, layer_info, \
                                             base_dir, timeloop_dir, top_dir, sub_dir, \
                                             configuration_dict, mode="search", \
                                             prev_block_info_dict=current_block_info_dict, subset_layers=subset_layers, \
                                             prev_cost_dict=current_cost_dict, prev_rehash_cost_dict=current_rehash_cost_dict)
            
        else:
            current_cost_dict, current_rehash_cost_dict, current_block_info_dict = \
            PartialUpdateAuthBlockAssignment(n_layers, layer_info, \
                                             base_dir, timeloop_dir, top_dir, sub_dir, \
                                             configuration_dict, \
                                             mode="search", \
                                             prev_block_info_dict=None, subset_layers=[], \
                                             prev_cost_dict=current_cost_dict, prev_rehash_cost_dict=None)
            
        solution_energy, solution_latency, solution_add_mem_traffic = 0, 0, 0
        for key in solution_cost_dict:
            solution_energy += solution_cost_dict[key]['total_energy'] / 10**6
            solution_latency += solution_cost_dict[key]['total_latency']
            solution_add_mem_traffic += solution_cost_dict[key]['add_memory_traffic']
        for key in solution_rehash_cost_dict:
            solution_energy += solution_rehash_cost_dict[key]['total_energy'] / 10**6
            solution_latency += solution_rehash_cost_dict[key]['total_latency']
            solution_add_mem_traffic += solution_rehash_cost_dict[key]['add_memory_traffic']
        
        current_energy, current_latency, current_add_mem_traffic = 0, 0, 0
        for key in current_cost_dict:
            current_energy += current_cost_dict[key]['total_energy'] / 10**6
            current_latency += current_cost_dict[key]['total_latency']
            current_add_mem_traffic += current_cost_dict[key]['add_memory_traffic']
        for key in current_rehash_cost_dict:
            current_energy += current_rehash_cost_dict[key]['total_energy'] / 10**6
            current_latency += current_rehash_cost_dict[key]['total_latency']
            current_add_mem_traffic += current_rehash_cost_dict[key]['add_memory_traffic']
        
        cost_solution = solution_energy * solution_latency
        cost_current = current_energy * current_latency
        cost_diff = (cost_solution - cost_current) / (10 ** 6 * n_layers)
        
        if cost_current < cost_best:
            best_state = copy.deepcopy(current_state)
            cost_best = cost_current
            print("Found best so far: ", best_state, " .. updating cost_best: {}".format(cost_best))
            
        if cost_diff > 0 or (random.uniform(0, 1) < math.exp(cost_diff / current_temp)):
            solution_state = copy.deepcopy(current_state)
            solution_cost_dict = copy.deepcopy(current_cost_dict)
            solution_rehash_cost_dict = copy.deepcopy(current_rehash_cost_dict)
            solution_block_info_dict = copy.deepcopy(current_block_info_dict)
        else:
            # roll-back to the solution state
            xml_file = os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, "layer{}".format(layer2change), \
                                      "eval{}".format(solution_state[layer2change - 1]), "timeloop-model.map+stats.xml")
            shutil.copy(xml_file, os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'layer{}'.format(layer2change)))
            current_state = copy.deepcopy(solution_state)
            current_cost_dict = copy.deepcopy(solution_cost_dict)
            current_rehash_cost_dict = copy.deepcopy(solution_rehash_cost_dict)
            current_block_info_dict = copy.deepcopy(solution_block_info_dict)
        
        solution_energy, solution_latency, solution_add_mem_traffic = 0, 0, 0
        for key in solution_cost_dict:
            solution_energy += solution_cost_dict[key]['total_energy'] / 10**6
            solution_latency += solution_cost_dict[key]['total_latency']
            solution_add_mem_traffic += solution_cost_dict[key]['add_memory_traffic']
        for key in solution_rehash_cost_dict:
            solution_energy += solution_rehash_cost_dict[key]['total_energy'] / 10**6
            solution_latency += solution_rehash_cost_dict[key]['total_latency']
            solution_add_mem_traffic += solution_rehash_cost_dict[key]['add_memory_traffic']
            
        # print("Solution state: ", solution_state)
        print("Current iteration: {} (temperature: {:.2f}) -- Latency: {} ({:.2f}% faster), Energy: {} uW ({:.2f}% lower), Add Mem Traffic: {} bits ({:.2f}% smaller)"\
              .format(i+1, current_temp, solution_latency, (baseline_latency - solution_latency) / float(baseline_latency) * 100. , \
                      solution_energy, (baseline_energy - solution_energy) / baseline_energy * 100., \
                      solution_add_mem_traffic, (baseline_add_mem_traffic - solution_add_mem_traffic) / float(baseline_add_mem_traffic) * 100.))
    
        curr_log = [(i + 1), current_temp, cost_solution, solution_latency, solution_energy, solution_add_mem_traffic]
        logs.append(curr_log)
        i += 1
        
        if current_temp < final_temp:
            break
            
    # print("Execution time: {}s".format(time.time() - start_time))
    
    # dump to csv file
    with open(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'SA_{}_top{}_summary.csv'.format(cooling_scheduler, topk)), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(logs)
        
    # dump best state & solution state to yaml file
    state = {'best': best_state, 'final': solution_state}
    with open(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'SA_{}_state.yaml'.format(cooling_scheduler)), 'w') as f:
        _ = yaml.dump(state, f)

    with open(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'SA_{}_state.yaml'.format('linear')), 'r') as f:
        states = yaml.safe_load(f)
        best_state = states['best']
    
    # move the best solution result
    for layer_idx in range(1, n_layers + 1):
        loopnest_id = best_state[layer_idx - 1]
        src = os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'layer{}'.format(layer_idx), \
                           'eval{}'.format(loopnest_id))
        src_files = os.listdir(src)
        for file in src_files:
            file_name = os.path.join(src, file)
            if os.path.isfile(file_name):
                shutil.copy(file_name, os.path.join(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir, 'layer{}'.format(layer_idx))))

def check_timeloop_exists(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, topk):
    # check if timeloop results already exists in a given folder
    flag = True
    for layer_id in unique_layers:
        path = os.path.join(base_dir, timeloop_dir, 'scheduling', sub_dir, 'layer{}'.format(layer_id))
        # print(os.path.exists(path))
        for k in range(1, topk + 1):
            path_k = os.path.join(path, 'mapping{}.yaml'.format(k))
            # print(path_k, os.path.exists(path_k))
            if not os.path.isfile(path_k):
                flag = False
                break
    return flag
    
def main():
    # define options
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, help='a path to the architecture description folder (e.g., designs/eyeriss_like/ver0/)')
    parser.add_argument('--workload', choices=['alexnet', 'resnet18', 'mobilenet_v2'], required=True, help='a DNN workload: alexnet (first 5 conv layers), resnet18, and mobilenet_v2')
    parser.add_argument('--workload_batch_size', default=1, type=int, help='batch size for the workload (default: 1)')
    parser.add_argument('--scheduler', choices=['baseline-timeloop-only', 'crypt-tile-single', 'crypt-opt-single', 'crypt-opt-cross'], required=True, help='scheduler algorithm: base-tile-single, crypt-tile-single, crypt-opt-single, and crypt-opt-cross')
    parser.add_argument('--topk', type=int, default=6, help='k for the top-k loopnest search')
    parser.add_argument('--rerun_timeloop', default=False, action='store_true', help='rerun Timeloop search even if the results form the previous run are found')

    args = parser.parse_args()

    # define paths / workloads
    base_dir = Path(os.getcwd())
    timeloop_dir = args.arch
    top_dir = 'workloads'
    sub_dir = '{}_batch{}'.format(args.workload, args.workload_batch_size)
    
    if args.workload == 'alexnet':
        net = model_zoo.alexnet(pretrained=False)
        layers_exclude_from_search = [6, 7, 8]
    elif args.workload == 'resnet18':
        net = model_zoo.resnet18(pretrained=False)
        layers_exclude_from_search = []
    elif args.workload == 'mobilenet_v2':
        net = model_zoo.mobilenet_v2(pretrained=False)
        layers_exclude_from_search = []

    input_size = (3, 224, 224)
    batch_size = args.workload_batch_size

    # Convert to the timeloop workload
    if not os.path.exists(os.path.join(top_dir, sub_dir)):
        pytorch2timeloop.convert_model(
                net,
                input_size,
                batch_size,
                sub_dir,
                top_dir,
                True,
                exception_module_names
            )

    # Extract layer info
    n_layers, unique_layers, layer_info, layer_info_ignore_interlayer = extract_layer_info(net, input_size, base_dir, top_dir, sub_dir)

    # Configure the top-k parameter in the mapper.yaml file
    mapper_file_path = os.path.join(base_dir, timeloop_dir, 'mapper/mapper.yaml')
    with open(mapper_file_path, 'r') as f:
        mapper_config = yaml.safe_load(f)
    mapper_config['mapper']['topk'] = args.topk
    with open(mapper_file_path, 'w') as f:
        _ = yaml.dump(mapper_config, f)

    # Load configuration dict
    with open(os.path.join(timeloop_dir, 'config.yaml'), 'r') as f:
        configuration_dict = yaml.safe_load(f)

    # Run timeloop
    timeloop_exists = check_timeloop_exists(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, args.topk)

    if args.rerun_timeloop or not timeloop_exists:
        run_timeloop(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, args.topk, base=(args.scheduler=='baseline-timeloop-only'))
        run_timeloop_model(base_dir, timeloop_dir, top_dir, sub_dir, unique_layers, base=(args.scheduler=='baseline-timeloop-only'))

    if args.scheduler == 'baseline-timeloop-only':
        return

    # Run simulated annealing if the scheduler is 'crypt-opt-cross'
    if args.scheduler == 'crypt-opt-cross':
        if not os.path.exists(os.path.join(base_dir, timeloop_dir, 'joint_topk')):
            os.mkdir(os.path.join(base_dir, timeloop_dir, 'joint_topk'))
        if not os.path.exists(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir)):
            os.mkdir(os.path.join(base_dir, timeloop_dir, 'joint_topk', sub_dir))
        run_simulated_annealing(n_layers, layer_info, base_dir, timeloop_dir, top_dir, sub_dir, configuration_dict, args.topk, layers_exclude_from_search, \
                                args.workload)

    # Finally, get the AuthBlock assignment result and save it
    if args.scheduler == 'crypt-tile-single':
        _layer_info = layer_info_ignore_interlayer
        mode = 'tile'
        joint = False
        evaluation_folder = 'evaluation'
    elif args.scheduler == 'crypt-opt-single':
        _layer_info = layer_info
        mode = 'search'
        joint = False
        evaluation_folder = 'evaluation'
    elif args.scheduler == 'crypt-opt-cross':
        _layer_info = layer_info
        mode = 'search'
        joint = True
        evaluation_folder = 'joint_topk'
        
    cost_dict, rehash_cost_dict, block_info_dict = \
    AuthBlockAssignment(n_layers, _layer_info, \
                        base_dir, timeloop_dir, top_dir, sub_dir, \
                        configuration_dict, mode=mode, \
                        joint=joint, generate_summary=True, return_cost_dict=True)

    dump_dst = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, '{}_cost.yaml'.format(args.scheduler))
    with open(dump_dst, 'w') as f:
        _ = yaml.dump({'cost_dict': cost_dict, 'rehash_cost_dict': rehash_cost_dict, 'block_info_dict': block_info_dict}, f)

if __name__ == '__main__':
    main()

    