import csv
import os
import yaml
import shutil
from pathlib import Path

from testbench_utils import generate_memory_traffic_dict, generate_rehash_info_dict, get_action_dict, get_action_dict_for_rehash
from summary_writer import write_summary

def configure_option(mode, n_layers, authblock_size=-1, authblock_orientation=None):
    search_block_size = {}
    predefined_u = {}
    predefined_perm = {}

    if mode == "fixed":
        assert (authblock_size > 1 and authblock_orientation != None)
        search_flag = False
        predefined_u_value = authblock_size
        predefined_perm_value = authblock_orientation

    elif mode == "tile":
        search_flag = False
        predefined_u_value = 'tile'
        predefined_perm_value = None

    elif mode == "search":
        search_flag = True
        predefined_u_value = 'tile'
        predefined_perm_value = None

    for layer_id in range(1, n_layers+1):
        search_block_size[layer_id] = search_flag
        predefined_u[layer_id] = {}
        predefined_perm[layer_id] = {}
        predefined_u[layer_id]['W'] = predefined_u_value
        predefined_u[layer_id]['I'] = predefined_u_value if mode != "search" else -1
        predefined_u[layer_id]['O'] = predefined_u_value
        predefined_perm[layer_id]['W'] = predefined_perm_value
        predefined_perm[layer_id]['I'] = predefined_perm_value
        predefined_perm[layer_id]['O'] = predefined_perm_value
        
    return search_block_size, predefined_u, predefined_perm
    
def get_latency_energy_info(configuration_dict):
    # Fully pipelined AES-GCM: 1 cycle / 1 cycle
    if configuration_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'] == 1:
        AESGCM_energy_profile = {'AES': 1.29 * 128, 'GCM': 57.7, 'XOR': 0} # pJ / op (e.g., 128-bit AES, 128-bit * 128-bit GCM)
        AESGCM_latency_profile = {'AES': 1, 'GCM': 1, 'XOR': 1} # cycle / op 

    # Parallel AES-GCM: 11 cycle / 8 cycle
    elif configuration_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'] == 11:
        AESGCM_energy_profile = {'AES': 1.52 * 128, 'GCM': 82.4, 'XOR': 0} # pJ / op (e.g., 128-bit AES, 128-bit * 128-bit GCM)
        AESGCM_latency_profile = {'AES': 11, 'GCM': 8, 'XOR': 1} # cycle / op 

    # Serial AES-GCM: 336 cycle / 128 cycle
    elif configuration_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'] == 336:
        AESGCM_energy_profile = {'AES': 6 * 128, 'GCM': 345.6, 'XOR': 0} # pJ / op (e.g., 128-bit AES, 128-bit * 128-bit GCM)
        AESGCM_latency_profile = {'AES': 336, 'GCM': 128, 'XOR': 1} # cycle / op 
        
    return AESGCM_energy_profile, AESGCM_latency_profile
    
def generate_cost_dict(n_layers, configuration_dict, cycle_dict, energy_dict, \
                       memory_traffic_dict, block_info_dict, rehash_info_dict, \
                       cryptographic_action_count_dict, rehash_action_count_dict, \
                       AESGCM_energy_profile, AESGCM_latency_profile, DRAM_read_per_bit_energy, DRAM_write_per_bit_energy, \
                       partial_update=False, cost_dict_given={}, partial_update_layer_list=[]):
    
    if partial_update:
        assert (bool(cost_dict_given))
        cost_dict = cost_dict_given
        layers_to_update = partial_update_layer_list
    else:
        cost_dict = {}
        for layer_id in range(1, n_layers + 1):
            cost_dict[layer_id] = {}
        layers_to_update = list(range(1, n_layers + 1))
        
    for layer_id in layers_to_update:
        
        # cost_dict[layer_id] = {}

        total_memory_read_bits = 0
        total_memory_write_bits = 0
        additional_read_bits = 0
        additional_write_bits = 0
        total_redundant_bits = 0
        total_hash_bits = 0

        aes_counts = []
        gcm_counts = []
        xor_counts = []
        
        # weights
        weight_memory_traffic_dict = memory_traffic_dict[layer_id]['W']
        weight_block_info_dict = block_info_dict[layer_id]['W']
        weight_action_count_dict = cryptographic_action_count_dict[layer_id]['W']

        total_memory_read_bits += weight_action_count_dict['total_read_bits']
        total_memory_write_bits += weight_action_count_dict['total_write_bits']

        additional_read_bits +=  weight_action_count_dict['additional_read_bits']
        additional_write_bits += weight_action_count_dict['additional_write_bits']
        
        total_redundant_bits += weight_action_count_dict['redundant_bits']
        total_hash_bits += weight_action_count_dict['tag_bits']

        aes_counts.append(weight_action_count_dict['aes_engine_count'])
        gcm_counts.append(weight_action_count_dict['gf_mult_count'])
        xor_counts.append(weight_action_count_dict['xor_count'])

        # inputs
        input_memory_traffic_dict = memory_traffic_dict[layer_id]['I']
        input_block_info_dict = block_info_dict[layer_id]['I']
        input_action_count_dict = cryptographic_action_count_dict[layer_id]['I']

        total_memory_read_bits += input_action_count_dict['total_read_bits']
        total_memory_write_bits += input_action_count_dict['total_write_bits']

        additional_read_bits += input_action_count_dict['additional_read_bits']
        additional_write_bits += input_action_count_dict['additional_write_bits']
        
        total_redundant_bits += input_action_count_dict['redundant_bits']
        total_hash_bits += input_action_count_dict['tag_bits']

        aes_counts.append(input_action_count_dict['aes_engine_count'])
        gcm_counts.append(input_action_count_dict['gf_mult_count'])
        xor_counts.append(input_action_count_dict['xor_count'])

        # outputs
        output_memory_traffic_dict = memory_traffic_dict[layer_id]['O']
        output_block_info_dict = block_info_dict[layer_id]['O']
        output_action_count_dict = cryptographic_action_count_dict[layer_id]['O']

        total_memory_read_bits += output_action_count_dict['total_read_bits']
        total_memory_write_bits += output_action_count_dict['total_write_bits']

        additional_read_bits += output_action_count_dict['additional_read_bits']
        additional_write_bits += output_action_count_dict['additional_write_bits']
        
        total_redundant_bits += output_action_count_dict['redundant_bits']
        total_hash_bits += output_action_count_dict['tag_bits']

        aes_counts.append(output_action_count_dict['aes_engine_count'])
        gcm_counts.append(output_action_count_dict['gf_mult_count'])
        xor_counts.append(output_action_count_dict['xor_count'])
        
        # baseline energy / latency
        cycle = cycle_dict[layer_id]
        energy = energy_dict[layer_id]
        
        if configuration_dict['CRYPT_ENGINE_SHARED']:
            aes_latency = sum(aes_counts) * (AESGCM_latency_profile['AES'] + AESGCM_latency_profile['XOR']) \
                          / sum(configuration_dict['CRYPT_ENGINE_COUNT'])
            gcm_latency = sum(gcm_counts) * (AESGCM_latency_profile['GCM'] + AESGCM_latency_profile['XOR']) \
                          / sum(configuration_dict['CRYPT_ENGINE_COUNT'])
            crypt_latency = max(aes_latency, gcm_latency) # assuming AES and GF can be pipelined
        else:
            aes_latency = [aes_counts[i] * (AESGCM_latency_profile['AES'] + AESGCM_latency_profile['XOR']) \
                           / (configuration_dict['CRYPT_ENGINE_COUNT'][i]) for i in range(3)]
            gcm_latency = [gcm_counts[i] * (AESGCM_latency_profile['GCM'] + AESGCM_latency_profile['XOR']) \
                           / (configuration_dict['CRYPT_ENGINE_COUNT'][i]) for i in range(3)]
            crypt_latency = max([max(aes_latency[i], gcm_latency[i]) for i in range(3)])
            
        memory_latency = max(total_memory_read_bits / (configuration_dict['DRAM_READ_BANDWIDTH'] * configuration_dict['WORDBITS']), \
                             total_memory_write_bits / (configuration_dict['DRAM_WRITE_BANDWIDTH'] * configuration_dict['WORDBITS']))
        total_latency = max(cycle, crypt_latency, memory_latency)
        
        aes_energy = sum(aes_counts) * AESGCM_energy_profile['AES']
        gcm_energy = sum(gcm_counts) * AESGCM_energy_profile['GCM']
        xor_energy = sum(xor_counts) * AESGCM_energy_profile['XOR']
        
        memory_energy = additional_read_bits * DRAM_read_per_bit_energy + additional_write_bits * DRAM_write_per_bit_energy
        total_energy = energy + (aes_energy + gcm_energy + xor_energy) + memory_energy

        cost_dict[layer_id]['timeloop_cycle'] = cycle
        cost_dict[layer_id]['timeloop_energy'] = energy
        cost_dict[layer_id]['crypt_latency'] = crypt_latency
        cost_dict[layer_id]['memory_latency'] = memory_latency
        cost_dict[layer_id]['total_latency'] = total_latency
        cost_dict[layer_id]['crypt_energy'] = (aes_energy + gcm_energy + xor_energy)
        cost_dict[layer_id]['add_memory_energy'] = memory_energy
        cost_dict[layer_id]['total_energy'] = total_energy
        cost_dict[layer_id]['add_memory_traffic'] = additional_read_bits + additional_write_bits
        cost_dict[layer_id]['total_redundant_bits'] = total_redundant_bits
        cost_dict[layer_id]['total_hash_bits'] = total_hash_bits
        
    rehash_cost_dict = {}
    for key in rehash_action_count_dict.keys():
        aes_latency = rehash_action_count_dict[key]['aes_engine_count'] * (AESGCM_latency_profile['AES'] + AESGCM_latency_profile['XOR']) \
                      / (configuration_dict['CRYPT_ENGINE_COUNT'][1] + configuration_dict['CRYPT_ENGINE_COUNT'][2])
        gcm_latency = rehash_action_count_dict[key]['gf_mult_count'] * (AESGCM_latency_profile['GCM'] + AESGCM_latency_profile['XOR']) \
                      / (configuration_dict['CRYPT_ENGINE_COUNT'][1] + configuration_dict['CRYPT_ENGINE_COUNT'][2])
        crypt_latency = max(aes_latency, gcm_latency)

        aes_energy = rehash_action_count_dict[key]['aes_engine_count'] * AESGCM_energy_profile['AES']
        gcm_energy = rehash_action_count_dict[key]['gf_mult_count'] * AESGCM_energy_profile['GCM']
        xor_energy = rehash_action_count_dict[key]['xor_count'] * AESGCM_energy_profile['XOR']

        memory_latency = max(rehash_action_count_dict[key]['total_read_bits'] / \
                             (configuration_dict['DRAM_READ_BANDWIDTH'] * configuration_dict['WORDBITS']), \
                             rehash_action_count_dict[key]['total_write_bits'] / \
                             (configuration_dict['DRAM_WRITE_BANDWIDTH'] * configuration_dict['WORDBITS']))
        memory_energy = rehash_action_count_dict[key]['total_read_bits'] * DRAM_read_per_bit_energy + \
                        rehash_action_count_dict[key]['total_write_bits'] * DRAM_write_per_bit_energy

        total_latency = max(crypt_latency, memory_latency)
        total_energy = (aes_energy + gcm_energy + xor_energy) + memory_energy

        rehash_cost_dict[key] = {}
        rehash_cost_dict[key]['crypt_latency'] = crypt_latency
        rehash_cost_dict[key]['memory_latency'] = memory_latency
        rehash_cost_dict[key]['total_latency'] = total_latency
        rehash_cost_dict[key]['crypt_energy'] = (aes_energy + gcm_energy + xor_energy)
        rehash_cost_dict[key]['add_memory_energy'] = memory_energy
        rehash_cost_dict[key]['total_energy'] = total_energy
        rehash_cost_dict[key]['add_memory_traffic'] = (rehash_action_count_dict[key]['total_read_bits'] + \
                                                       rehash_action_count_dict[key]['total_write_bits'])
        
    return cost_dict, rehash_cost_dict
    
def AuthBlockAssignment(n_layers, layer_info, \
                        base_dir, timeloop_dir, top_dir, sub_dir, \
                        configuration_dict, \
                        mode="search", authblock_size=-1, authblock_orientation=None, \
                        joint=False, generate_summary=False, return_cost_dict=False):
    
    WORD_SIZE = configuration_dict['WORDBITS']
    TAG_SIZE = 64
    
    search_block_size, predefined_u, predefined_perm = configure_option(mode, n_layers, authblock_size, authblock_orientation)
    
    memory_traffic_dict, block_info_dict = generate_memory_traffic_dict(n_layers, layer_info, predefined_u, predefined_perm, \
                                                                        base_dir, timeloop_dir, top_dir, sub_dir, \
                                                                        search_block_size, u_multiple_of=128//WORD_SIZE, \
                                                                        WORD_SIZE=WORD_SIZE, TAG_SIZE=TAG_SIZE, use_joint_topk=joint)
    
    
    rehash_info_dict = generate_rehash_info_dict(n_layers, layer_info, block_info_dict, \
                                                 base_dir, timeloop_dir, top_dir, sub_dir, use_joint_topk=joint)
    
    cryptographic_action_count_dict = get_action_dict(n_layers, memory_traffic_dict, block_info_dict, \
                                                      WORD_SIZE, TAG_SIZE, AES_DATAPATH=128)
    rehash_action_count_dict = get_action_dict_for_rehash(rehash_info_dict, block_info_dict, WORD_SIZE, TAG_SIZE, AES_DATAPATH=128)

    AESGCM_energy_profile, AESGCM_latency_profile = get_latency_energy_info(configuration_dict)
    
    # TODO: this is only for LPDDR4; add support for other DRAM models without having to hard-code it
    MEMORY_READ_PER_BIT_ENERGY = 8
    MEMORY_WRITE_PER_BIT_ENERGY = 8
    
    if generate_summary:
        write_summary(n_layers, layer_info, \
                      base_dir, timeloop_dir, top_dir, sub_dir, \
                      configuration_dict, \
                      memory_traffic_dict, block_info_dict, rehash_info_dict, \
                      cryptographic_action_count_dict, rehash_action_count_dict, \
                      AESGCM_energy_profile, AESGCM_latency_profile, MEMORY_READ_PER_BIT_ENERGY, MEMORY_WRITE_PER_BIT_ENERGY, \
                      joint=joint, filename="{}_{}.csv".format(mode, 'joint' if joint else 'per_layer'))
        
    if return_cost_dict:
        
        # create timeloop baseline energy/cycle dict
        cycle_dict = {}
        energy_dict = {}
        
        evaluation_folder = 'evaluation' if not joint else 'joint_topk'
        
        for layer_id in range(1, n_layers + 1):
            summary = [layer_id]
            layer_id_for_timeloop = layer_id if joint else layer_info[layer_id]['layer_id_for_timeloop']
            stats_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                      "timeloop-model.stats.txt")
            with open(stats_file, 'r') as f:
                lines = f.read().split('\n')[-200:]
                for line in lines:
                    if line.startswith('Energy'):
                        energy = eval(line.split(': ')[1].split(' ')[0]) * float(10**6) # micro to pico
                    elif line.startswith('Cycles'):
                        cycle = eval(line.split(': ')[1])
            
            cycle_dict[layer_id] = cycle
            energy_dict[layer_id] = energy
        
        cost_dict, rehash_cost_dict = generate_cost_dict(n_layers, configuration_dict, cycle_dict, energy_dict, \
                                                         memory_traffic_dict, block_info_dict, rehash_info_dict, \
                                                         cryptographic_action_count_dict, rehash_action_count_dict, \
                                                         AESGCM_energy_profile, AESGCM_latency_profile, MEMORY_READ_PER_BIT_ENERGY, \
                                                         MEMORY_WRITE_PER_BIT_ENERGY)
        
        return cost_dict, rehash_cost_dict, block_info_dict
    
    
# Partial update only affected layers - speed up evaluation for simulated annealing
def PartialUpdateAuthBlockAssignment(n_layers, layer_info, \
                                     base_dir, timeloop_dir, top_dir, sub_dir, \
                                     configuration_dict, \
                                     mode="search", authblock_size=-1, authblock_orientation=None, \
                                     prev_block_info_dict=None, subset_layers=[], \
                                     prev_cost_dict=None, prev_rehash_cost_dict=None):
    
    WORD_SIZE = configuration_dict['WORDBITS']
    TAG_SIZE = 64
    
    search_block_size, predefined_u, predefined_perm = configure_option(mode, n_layers, authblock_size, authblock_orientation)
    
    memory_traffic_dict, block_info_dict = generate_memory_traffic_dict(n_layers, layer_info, predefined_u, predefined_perm, \
                                                                        base_dir, timeloop_dir, top_dir, sub_dir, \
                                                                        search_block_size, u_multiple_of=128//WORD_SIZE, \
                                                                        WORD_SIZE=WORD_SIZE, TAG_SIZE=TAG_SIZE, use_joint_topk=True, \
                                                                        use_subset=subset_layers, prev_block_info_dict=prev_block_info_dict)
    
    
    rehash_info_dict = generate_rehash_info_dict(n_layers, layer_info, block_info_dict, \
                                                 base_dir, timeloop_dir, top_dir, sub_dir, use_joint_topk=True)
    
    cryptographic_action_count_dict = get_action_dict(n_layers, memory_traffic_dict, block_info_dict, \
                                                      WORD_SIZE, TAG_SIZE, AES_DATAPATH=128)
    rehash_action_count_dict = get_action_dict_for_rehash(rehash_info_dict, block_info_dict, WORD_SIZE, TAG_SIZE, AES_DATAPATH=128)

    AESGCM_energy_profile, AESGCM_latency_profile = get_latency_energy_info(configuration_dict)
    
    # TODO: this is only for LPDDR4; add support for other DRAM models without having to hard-code it
    MEMORY_READ_PER_BIT_ENERGY = 8
    MEMORY_WRITE_PER_BIT_ENERGY = 8
        
    # create timeloop baseline energy/cycle dict
    cycle_dict = {}
    energy_dict = {}

    evaluation_folder = 'joint_topk'

    for layer_id in range(1, n_layers + 1):
        cycle_dict[layer_id] = prev_cost_dict[layer_id]['timeloop_cycle']
        energy_dict[layer_id] = prev_cost_dict[layer_id]['timeloop_energy']

    cost_dict, rehash_cost_dict = generate_cost_dict(n_layers, configuration_dict, cycle_dict, energy_dict, \
                                                     memory_traffic_dict, block_info_dict, rehash_info_dict, \
                                                     cryptographic_action_count_dict, rehash_action_count_dict, \
                                                     AESGCM_energy_profile, AESGCM_latency_profile, MEMORY_READ_PER_BIT_ENERGY, \
                                                     MEMORY_WRITE_PER_BIT_ENERGY, \
                                                     partial_update=True, cost_dict_given=prev_cost_dict, partial_update_layer_list=subset_layers)

    return cost_dict, rehash_cost_dict, block_info_dict
    
    
    
    