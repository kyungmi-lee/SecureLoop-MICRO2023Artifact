import csv
import os
import yaml
import shutil
from pathlib import Path

def write_summary(n_layers, layer_info, \
                  base_dir, timeloop_dir, top_dir, sub_dir, \
                  configuration_dict, \
                  memory_traffic_dict, block_info_dict, rehash_info_dict, \
                  cryptographic_action_count_dict, rehash_action_count_dict, \
                  AESGCM_energy_profile, AESGCM_latency_profile, DRAM_read_per_bit_energy, DRAM_write_per_bit_energy, \
                  joint=False, filename=None):
    
    evaluation_folder = 'evaluation' if not joint else 'joint_topk'

    summaries = []
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

        summary.extend([cycle, energy])

        total_memory_read_bits = 0
        total_memory_write_bits = 0
        additional_read_bits = 0
        additional_write_bits = 0
        tag_bits = 0
        redundant_bits = 0

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

        tag_bits += weight_action_count_dict['tag_bits']
        redundant_bits += weight_action_count_dict['redundant_bits']

        # aes counts, gcm_counts, xor_counts, additional_read_bits, additional_write_bits, u
        summary.extend([weight_action_count_dict['aes_engine_count'], \
                        weight_action_count_dict['gf_mult_count'], \
                        weight_action_count_dict['xor_count'], \
                        weight_action_count_dict['additional_read_bits'], \
                        weight_action_count_dict['additional_write_bits'], \
                        weight_block_info_dict['u'], \
                        weight_action_count_dict['tag_bits'], \
                        weight_action_count_dict['redundant_bits']])

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

        tag_bits += input_action_count_dict['tag_bits']
        redundant_bits += input_action_count_dict['redundant_bits']

        # aes counts, gcm_counts, xor_counts, additional_read_bits, additional_write_bits, u
        summary.extend([input_action_count_dict['aes_engine_count'], \
                        input_action_count_dict['gf_mult_count'], \
                        input_action_count_dict['xor_count'], \
                        input_action_count_dict['additional_read_bits'], \
                        input_action_count_dict['additional_write_bits'], \
                        input_block_info_dict['u'], \
                        input_block_info_dict['permutation'], \
                        input_action_count_dict['tag_bits'], \
                        input_action_count_dict['redundant_bits']])

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

        tag_bits += output_action_count_dict['tag_bits']
        redundant_bits += output_action_count_dict['redundant_bits']

        # aes counts, gcm_counts, xor_counts, additional_read_bits, additional_write_bits, u
        summary.extend([output_action_count_dict['aes_engine_count'], \
                        output_action_count_dict['gf_mult_count'], \
                        output_action_count_dict['xor_count'], \
                        output_action_count_dict['additional_read_bits'], \
                        output_action_count_dict['additional_write_bits'], \
                        output_block_info_dict['u'], \
                        output_action_count_dict['tag_bits'], \
                        output_action_count_dict['redundant_bits']])

        aes_counts.append(output_action_count_dict['aes_engine_count'])
        gcm_counts.append(output_action_count_dict['gf_mult_count'])
        xor_counts.append(output_action_count_dict['xor_count'])

        # get crypto-latency for this layer
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

        summary.extend([crypt_latency, memory_latency, total_latency])

        # get crypto-energy for this layer
        aes_energy = sum(aes_counts) * AESGCM_energy_profile['AES']
        gcm_energy = sum(gcm_counts) * AESGCM_energy_profile['GCM']
        xor_energy = sum(xor_counts) * AESGCM_energy_profile['XOR']

        memory_energy = additional_read_bits * DRAM_read_per_bit_energy + additional_write_bits * DRAM_write_per_bit_energy
        total_energy = energy + (aes_energy + gcm_energy + xor_energy) + memory_energy

        additional_mem_traffic = additional_read_bits + additional_write_bits

        summary.extend([(aes_energy + gcm_energy + xor_energy), memory_energy, total_energy])
        summary.extend([additional_mem_traffic, tag_bits, redundant_bits])

        summaries.append(summary)

        # print(key, additional_mem_traffic)

    for key in rehash_action_count_dict.keys():
        summary = ["Rehash{}-{}".format(key[0], key[1])]
        summary.extend([0, 0, \
                        0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, 0, 0, 0, 0, \
                        0, 0, 0, 0, 0, 0, 0, 0])

        # when there are non-shared AES-GCM engine
        # rehashing only for ifmap - ofmap
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

        additional_mem_traffic = rehash_action_count_dict[key]['total_read_bits'] + \
                                 rehash_action_count_dict[key]['total_write_bits']

        tag_bits = rehash_action_count_dict[key]['additional_read_bits'] + \
                   rehash_action_count_dict[key]['additional_write_bits']

        total_latency = max(crypt_latency, memory_latency)
        total_energy = (aes_energy + gcm_energy + xor_energy) + memory_energy

        summary.extend([crypt_latency, memory_latency, total_latency, \
                        (aes_energy + gcm_energy + xor_energy), memory_energy, total_energy])
        summary.extend([additional_mem_traffic, tag_bits, 0])

        summaries.append(summary)

    summary_header = ['Layer#', 'Baseline Cycle', 'Baseline Energy (pJ)', \
                      'Weight AES Count', 'Weight GFMult Count', 'Weight XOR Count', \
                      'Weight Additional Memory Read (bits)', 'Weight Additional Memory Write (bits)', \
                      'Weight Authentication Block Size', 'Weight Tag Bits', 'Weight Redundant Bits', \
                      'Input AES Count', 'Input GFMult Count', 'Input XOR Count', \
                      'Input Additional Memory Read (bits)', 'Input Additional Memory Write (bits)', \
                      'Input Authentication Block Size', 'Input Authentication Permutation', \
                      'Input Tag Bits', 'Input Redundant Bits', \
                      'Output AES Count', 'Output GFMult Count', 'Output XOR Count', \
                      'Output Additional Memory Read (bits)', 'Output Additional Memory Write (bits)', \
                      'Output Authentication Block Size', 'Output Tag Bits', 'Output Redundant Bits', \
                      'CryptEngine Latency', 'Final Memory Read/Write Latency', 'Total Latency', \
                      'CryptEngine Energy (pJ)', 'Additional Memory Read/Write Energy (pJ)', 'Total Energy (pJ)', \
                      'Additional Memory Traffic (bits)', 'Tag Bits', 'Redundant Bits']

    if filename is None:
        filename = 'stat.csv'
    write_dst = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, filename)
    with open(write_dst, 'w') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(summary_header)
        for result in summaries:
            csv_file.writerow(result)