import yaml
import os
import copy

import numpy as np
import math
import xml.etree.ElementTree as ET

from collections import OrderedDict
from collections import Counter
from itertools import islice

def configure_wordbits(arch_dict, wordbits):
    for key in arch_dict.keys():
        if key == 'word-bits' or key == 'datawidth':
            arch_dict[key] = wordbits
        if isinstance(arch_dict[key], list):
            for subdict in arch_dict[key]:
                configure_wordbits(subdict, wordbits)
        if isinstance(arch_dict[key], dict):
            configure_wordbits(arch_dict[key], wordbits)
            
def configure_dram_bandwidth(arch_dict, read_bandwidth, write_bandwidth):
    if 'name' in arch_dict.keys() and arch_dict['name'] == 'DRAM':
        arch_dict['attributes']['read_bandwidth'] = read_bandwidth
        arch_dict['attributes']['write_bandwidth'] = write_bandwidth
        if 'block-size' in arch_dict['attributes'].keys():
            arch_dict['attributes']['block-size'] = int(arch_dict['attributes']['width'] / arch_dict['attributes']['word-bits'])
    else:
        for key in arch_dict.keys():
            if isinstance(arch_dict[key], list):
                for subdict in arch_dict[key]:
                    configure_dram_bandwidth(subdict, read_bandwidth, write_bandwidth)
            if isinstance(arch_dict[key], dict):
                configure_dram_bandwidth(arch_dict[key], read_bandwidth, write_bandwidth)
                
                
def configure_sram(arch_dict, shared, depth, width, banks, read_bandwidth, write_bandwidth):
    if shared:
        if 'name' in arch_dict.keys() and arch_dict['name'] == 'shared_glb':
            arch_dict['attributes']['memory_depth'] = depth[0]
            arch_dict['attributes']['memory_width'] = width[0]
            arch_dict['attributes']['n_banks'] = banks[0]
            arch_dict['attributes']['read_bandwidth'] = read_bandwidth[0]
            arch_dict['attributes']['write_bandwidth'] = write_bandwidth[0]
            if 'block-size' in arch_dict['attributes'].keys():
                arch_dict['attributes']['block-size'] = int(arch_dict['attributes']['memory_width'] \
                                                            / arch_dict['attributes']['word-bits'])
        else:
            for key in arch_dict.keys():
                if isinstance(arch_dict[key], list):
                    for subdict in arch_dict[key]:
                        configure_sram(subdict, shared, depth, width, banks, read_bandwidth, write_bandwidth)
                if isinstance(arch_dict[key], dict):
                    configure_sram(arch_dict[key], shared, depth, width, banks, read_bandwidth, write_bandwidth)
    else:
        prefix = ['weights', 'ifmap', 'psum']
        for idx, p in enumerate(prefix):
            if 'name' in arch_dict.keys() and arch_dict['name'] == p + '_glb':
                arch_dict['attributes']['memory_depth'] = depth[idx]
                arch_dict['attributes']['memory_width'] = width[idx]
                arch_dict['attributes']['n_banks'] = banks[idx]
                arch_dict['attributes']['read_bandwidth'] = read_bandwidth[idx]
                arch_dict['attributes']['write_bandwidth'] = write_bandwidth[idx]
                if 'block-size' in arch_dict['attributes'].keys():
                    arch_dict['attributes']['block-size'] = int(arch_dict['attributes']['memory_width'] \
                                                                / arch_dict['attributes']['word-bits'])
            else:
                for key in arch_dict.keys():
                    if isinstance(arch_dict[key], list):
                        for subdict in arch_dict[key]:
                            configure_sram(subdict, shared, depth, width, banks, read_bandwidth, write_bandwidth)
                    if isinstance(arch_dict[key], dict):
                        configure_sram(arch_dict[key], shared, depth, width, banks, read_bandwidth, write_bandwidth)

def configure_pe_size(arch_dict, pe_x, pe_y):
    # find if meshX exists in attributes
    for key in arch_dict.keys():
        if key == 'attributes' and 'meshX' in arch_dict[key].keys():
            arch_dict[key]['meshX'] = pe_x
        else:
            if isinstance(arch_dict[key], list):
                for subdict in arch_dict[key]:
                    configure_pe_size(subdict, pe_x, pe_y)
            if isinstance(arch_dict[key], dict):
                configure_pe_size(arch_dict[key], pe_x, pe_y)
                
                
def adjust_spatial_size(arch_dict, pe_x, pe_y):
    # if [0..N] in 'name' then change N = pe_x * pe_y - 1 for PE,
    #                                 N = pe_x for DummyBuffer
    if 'name' in arch_dict.keys() and 'PE[0..' in arch_dict['name']:
        arch_dict['name'] = 'PE[0..' + str(int(pe_x * pe_y -1)) + ']'
    elif 'name' in arch_dict.keys() and 'DummyBuffer[0..' in arch_dict['name']:
        arch_dict['name'] = 'DummyBuffer[0..' + str(int(pe_x - 1)) + ']'
    else:
        for key in arch_dict.keys():
            if isinstance(arch_dict[key], list):
                for subdict in arch_dict[key]:
                    adjust_spatial_size(subdict, pe_x, pe_y)
            if isinstance(arch_dict[key], dict):
                adjust_spatial_size(arch_dict[key], pe_x, pe_y)
                
def configure_spad(arch_dict, shared, depth, width):
    if shared:
        if 'name' in arch_dict.keys() and arch_dict['name'] == 'pe_spad':
            arch_dict['attributes']['memory_depth'] = depth[0]
            arch_dict['attributes']['memory_width'] = width[0]
            if 'block-size' in arch_dict['attributes'].keys():
                arch_dict['attributes']['block-size'] = int(arch_dict['attributes']['memory_width'] \
                                                            / arch_dict['attributes']['word-bits'])
        else:
            for key in arch_dict.keys():
                if isinstance(arch_dict[key], list):
                    for subdict in arch_dict[key]:
                        configure_spad(subdict, shared, depth, width)
                if isinstance(arch_dict[key], dict):
                    configure_spad(arch_dict[key], shared, depth, width)
    else:
        prefix = ['weights', 'ifmap', 'psum']
        for idx, p in enumerate(prefix):
            if 'name' in arch_dict.keys() and arch_dict['name'] == p + '_spad':
                arch_dict['attributes']['memory_depth'] = depth[idx]
                arch_dict['attributes']['memory_width'] = width[idx]
                if 'block-size' in arch_dict['attributes'].keys():
                    arch_dict['attributes']['block-size'] = int(arch_dict['attributes']['memory_width'] \
                                                                / arch_dict['attributes']['word-bits'])
            else:
                for key in arch_dict.keys():
                    if isinstance(arch_dict[key], list):
                        for subdict in arch_dict[key]:
                            configure_spad(subdict, shared, depth, width)
                    if isinstance(arch_dict[key], dict):
                        configure_spad(arch_dict[key], shared, depth, width)
                        
def configure_effective_dram_model(arch_dict, dram_read_bandwidth, dram_write_bandwidth, wordbits, \
                                   crypt_engine_type, crypt_engine_cycle_per_block, crypt_engine_shared, crypt_engine_count, \
                                   conservative):
    if 'name' in arch_dict.keys() and arch_dict['name'] == 'DRAM':
        arch_dict['attributes']['technology'] = crypt_engine_type + '_{}cycle'.format(crypt_engine_cycle_per_block)
        
        effective_read_bandwidth = (128 / wordbits) / crypt_engine_cycle_per_block
        if not crypt_engine_shared:
            if conservative:
                effective_read_bandwidth *= min(crypt_engine_count[0], crypt_engine_count[1])
            else:
                effective_read_bandwidth *= (crypt_engine_count[0] + crypt_engine_count[1])
        else:
            effective_read_bandwidth *= crypt_engine_count[0] / 2
        effective_read_bandwidth = min(effective_read_bandwidth, dram_read_bandwidth)
        arch_dict['attributes']['read_bandwidth'] = effective_read_bandwidth
        
        effective_write_bandwidth = (128 / wordbits) / crypt_engine_cycle_per_block
        if not crypt_engine_shared:
            effective_write_bandwidth *= crypt_engine_count[2]
        else:
            effective_write_bandwidth *= crypt_engine_count[0] / 2
        effective_write_bandwidth = min(effective_write_bandwidth, dram_write_bandwidth)
        arch_dict['attributes']['write_bandwidth'] = effective_write_bandwidth
    else:
        for key in arch_dict.keys():
            if isinstance(arch_dict[key], list):
                for subdict in arch_dict[key]:
                    configure_effective_dram_model(subdict, dram_read_bandwidth, dram_write_bandwidth, wordbits, \
                                                   crypt_engine_type, crypt_engine_cycle_per_block, \
                                                   crypt_engine_shared, crypt_engine_count, conservative)
            if isinstance(arch_dict[key], dict):
                configure_effective_dram_model(arch_dict[key], dram_read_bandwidth, dram_write_bandwidth, wordbits, \
                                               crypt_engine_type, crypt_engine_cycle_per_block, \
                                               crypt_engine_shared, crypt_engine_count, conservative)
                
def generate_arch_files(arch_root, config_dict):
    with open(os.path.join(arch_root, 'template.yaml'), 'r') as f:
        template_arch = yaml.safe_load(f)
    arch_dict = copy.deepcopy(template_arch['architecture']['subtree'][0])
    configure_wordbits(arch_dict, config_dict['WORDBITS'])
    configure_dram_bandwidth(arch_dict, config_dict['DRAM_READ_BANDWIDTH'], config_dict['DRAM_WRITE_BANDWIDTH'])
    configure_sram(arch_dict, config_dict['SRAM_SHARED'], config_dict['SRAM_DEPTH'], config_dict['SRAM_WIDTH'], \
                   config_dict['SRAM_BANKS'], config_dict['SRAM_READ_BANDWIDTH'], config_dict['SRAM_WRITE_BANDWIDTH'])
    configure_pe_size(arch_dict, config_dict['PE_X'], config_dict['PE_Y'])
    adjust_spatial_size(arch_dict, config_dict['PE_X'], config_dict['PE_Y'])
    configure_spad(arch_dict, config_dict['PE_SPAD_SHARED'], config_dict['PE_SPAD_DEPTH'], config_dict['PE_SPAD_WIDTH'])

    baseline_file = {'architecture': {'version': 0.3, 'subtree': [arch_dict]}}
    baseline_file_name = 'baseline.yaml'
    with open(os.path.join(arch_root, 'baseline.yaml'), 'w') as f:
        _ = yaml.dump(baseline_file, f)
        
    configure_effective_dram_model(arch_dict, config_dict['DRAM_READ_BANDWIDTH'], config_dict['DRAM_WRITE_BANDWIDTH'], config_dict['WORDBITS'], \
                                   config_dict['CRYPT_ENGINE_TYPE'], config_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'], \
                                   config_dict['CRYPT_ENGINE_SHARED'], config_dict['CRYPT_ENGINE_COUNT'], \
                                   config_dict['EFFECTIVE_CONSERVATIVE'])
    effective_model_file = {'architecture': {'version': 0.3, 'subtree': [arch_dict]}}
    # effective_model_file_name = '{}_{}cycle.yaml'.format(config_dict['CRYPT_ENGINE_TYPE'], config_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'])
    with open(os.path.join(arch_root, 'effective.yaml'), 'w') as f:
        _ = yaml.dump(effective_model_file, f)
    
def parse_workload_file(path):
    with open(path, 'r') as f:
        workload = yaml.safe_load(f)
    dimension_list = workload['problem']['shape']['dimensions']
    stride_h = workload['problem']['instance']['Hstride']
    stride_w = workload['problem']['instance']['Wstride']
    kernel_size_r = workload['problem']['instance']['R']
    kernel_size_s = workload['problem']['instance']['S']
    padding_h = workload['problem']['instance']['Hdilation']
    padding_w = workload['problem']['instance']['Wdilation']
    P = workload['problem']['instance']['P']
    Q = workload['problem']['instance']['Q']
    
    return dimension_list, (stride_h, stride_w), (kernel_size_r, kernel_size_s), (padding_h, padding_w), (P, Q) 

def get_dimension_idx(dimension_list, dimension_of_interest):
    idx_list = []
    for d in dimension_of_interest:
        idx_list.append(dimension_list.index(d))
    return idx_list
    
def extract_from_xml(xml_path, get_mem_size=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # loopnest
    topology = root.findall('engine')[0].findall('topology_')[0]
    total_levels = int(topology.findall('levels_')[0].findall('count')[0].text) - 1 # 1 level for arithmetic, we only count memory levels
    levels = topology.findall('levels_')[0].findall('item')
    loopnest_depth = 0
    loopnest = []
    names = []
    keeps = []
    if get_mem_size:
        mem_size = []
    for idx in range(len(levels)):
        level = levels[idx].findall('px')[0]
        subnest = level.findall('subnest_')
        if len(subnest) > 0:
            subnest_depth = int(subnest[0].findall('count')[0].text)
            loopnest_depth += subnest_depth
            loops = subnest[0].findall('item')
            curr_loopnest = []
            for i in range(subnest_depth):
                dimension = int(loops[i].findall('dimension')[0].text)
                start = int(loops[i].findall('start')[0].text)
                end = int(loops[i].findall('end')[0].text)
                stride = int(loops[i].findall('stride')[0].text)
                spacetime_dimension = int(loops[i].findall('spacetime_dimension')[0].text)
                curr_loopnest.append((dimension, start, end, stride, spacetime_dimension))
            loopnest.append(curr_loopnest)

            spec = level.findall('specs_')
            name = spec[0].find('name')[0].text
            names.append(name)
            
            if get_mem_size:
                try:
                    m_size = int(spec[0].find('size')[0].text)
                except:
                    m_size = -1
                mem_size.append(m_size)

            stat = level.findall('stats_')
            keep = stat[0].findall('keep')[0].findall('PerDataSpace')[0].findall('item')
            temp = []
            for k in keep:
                temp.append(int(k.text))
            keeps.append(temp)
            
    if get_mem_size:
        return loopnest, names, keeps, mem_size
    
    return loopnest, names, keeps


def xml2mapping(xml_path, workload_path, constraint_path, dw):
    loopnest, names, keeps = extract_from_xml(xml_path)
    
    dimension_list, (stride_h, stride_w), (kernel_size_r, kernel_size_s), (padding_h, padding_w), (P, Q) \
    = parse_workload_file(workload_path)

    dim_list = ['N', 'C', 'P', 'Q', 'R', 'S'] if dw else ['N', 'M', 'C', 'P', 'Q', 'R', 'S'] 
    idx_list = get_dimension_idx(dimension_list, dim_list)
    
    # get architecture constraint
    with open(constraint_path, 'r') as f:
        constraint = yaml.safe_load(f)
    # get the list of "names" with spatial constraints
    component_with_spatial_constraint = []
    spatial_constraint_info = []
    for c in constraint['architecture_constraints']['targets']:
        if c['type'] == 'spatial':
            component_with_spatial_constraint.append(c['target'])
            spatial_constraint_info.append(c)
    
    mapping_list = []
    for (loop, name, keep) in zip(loopnest, names, keeps):
        t_factors = [1] * len(dim_list) # N, M, C, P, Q, R, S
        t_perm = []
        s_factors = [1] * len(dim_list) # N, M, C, P, Q, R, S
        # there can be multiple subloops with the same spatial factor --> loop unrolling of multiple dimensions
        s_x = []
        s_y = []
        for subloop in loop:
            dim = subloop[0]
            end = subloop[2]
            spatial = subloop[4]
            if spatial == 0:
                t_factors[idx_list.index(dim)] *= end
                t_perm.append(idx_list.index(dim))
            elif spatial == 1:
                s_factors[idx_list.index(dim)] *= end
                s_x.append(idx_list.index(dim))
            elif spatial == 2:
                s_factors[idx_list.index(dim)] *= end
                s_y.append(idx_list.index(dim))

        factors = ""
        for d, i in zip(dim_list, t_factors):
            factors += "{}={} ".format(d, i)
        # print(factors)

        permutation = ""
        if len(t_perm) > 0:
            for p in t_perm:
                permutation += dim_list[p]
        for idx, d in enumerate(dim_list):
            if idx not in t_perm:
                permutation += d
        # print(permutation)
        mapping = {'target': name, 'type': 'temporal', 'factors': factors, 'permutation': permutation}
        mapping_list.append(mapping)

        if len(s_x) > 0 or len(s_y) > 0:
            factors_spatial = ""
            for d, i in zip(dim_list, s_factors):
                factors_spatial += "{}={} ".format(d, i)
            # print(factors_spatial)

            s_permutation = ""
            if len(s_x) > 0 and len(s_y) > 0:
                for idx, d in enumerate(dim_list):
                    if idx in s_x:
                        s_permutation += d
                for idx, d in enumerate(dim_list):
                    if idx in s_y:
                        s_permutation += d
                # s_permutation += dim_list[s_x]
                # s_permutation += dim_list[s_y]
                for idx, d in enumerate(dim_list):
                    if idx not in s_x and idx not in s_y:
                        s_permutation += d
                split = len(s_x)
            elif len(s_x) > 0 and len(s_y) == 0:
                for idx, d in enumerate(dim_list):
                    if idx in s_x:
                        s_permutation += d
                for idx, d in enumerate(dim_list):
                    if idx not in s_x:
                        s_permutation += d
                split = len(s_x)
            elif len(s_y) > 0 and len(s_x) == 0:
                found_split = False
                for idx, d in enumerate(dim_list):
                    s_permutation += d
                    if idx in s_y and not found_split:
                        split = idx
                        found_split = True
            # print(s_permutation, split)

            mapping = {'target': name, 'type': 'spatial', 'factors': factors_spatial, 'permutation': s_permutation, 'split': split}
            mapping_list.append(mapping)
            
        if name in component_with_spatial_constraint and (len(s_x) == 0 and len(s_y) == 0):
            factors_spatial = ""
            for d in dim_list:
                factors_spatial += "{}={} ".format(d, 1)
            for info in spatial_constraint_info:
                if info['target'] == name:
                    spatial_info = info
                    break
            mapping = {'target': name, 'type': 'spatial', 'factors': factors_spatial, 'permutation': spatial_info['permutation'], \
                       'split': spatial_info['split']}
            mapping_list.append(mapping)

        # identify whether bypass information should be added
        keep_type = []
        bypass_type = []

        if keep[0] == 1:
            keep_type.append("Weights")
        else:
            bypass_type.append("Weights")

        if keep[1] == 1:
            keep_type.append("Inputs")
        else:
            bypass_type.append("Inputs")

        if keep[2] == 1:
            keep_type.append("Outputs")
        else:
            bypass_type.append("Outputs")

        if len(bypass_type) > 0 and len(keep_type) > 0:
            keep_str = "["
            bypass_str = "["
            for k in keep_type:
                keep_str += "{}, ".format(k)
            keep_str = keep_str[:-2] + "]"
            for b in bypass_type:
                bypass_str += "{}, ".format(b)
            bypass_str = bypass_str[:-2] + "]"

            mapping = {'target': name, 'type': 'bypass', 'keep': keep_type, 'bypass': bypass_type}
            mapping_list.append(mapping)

        elif len(bypass_type) > 0:
            bypass_str = "["
            for b in bypass_type:
                bypass_str += "{}, ".format(b)
            bypass_str = bypass_str[:-2] + "]"

            mapping = {'target': name, 'type': 'bypass', 'bypass': bypass_type}
            mapping_list.append(mapping)
    
    return mapping_list
    