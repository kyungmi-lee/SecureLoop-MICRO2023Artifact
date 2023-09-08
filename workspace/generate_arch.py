import os
import yaml
import shutil
from pathlib import Path
import argparse

from utils import generate_arch_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_design', type=str, default='eyeriss_like', help='name of the template design used (e.g., eyeriss_like)')
    parser.add_argument('--word_bits', type=int, default=16, help='the number of bits in a word (assume same word size for input, output, and weight)')

    # DRAM bandwidth options
    parser.add_argument('--dram_read_bandwidth', type=int, default=32, help='the number of words per cycle for DRAM read')
    parser.add_argument('--dram_write_bandwidth', type=int, default=32, help='the number of words per cycle for DRAM write')

    # SRAM options
    parser.add_argument('--sram_shared', default=True, type=bool, help='if SRAM is shared among all three datatypes')
    parser.add_argument('--sram_depth', default=[2**13], nargs='*', help='SRAM depth (height). If the SRAM is not shared, give three numbers for each SRAM in order: weight, ifmap, ofmap')
    parser.add_argument('--sram_width', default=[2**7], nargs='*', help='SRAM width. If the SRAM is not shared, give three numbers for each SRAM in order: weight, ifmap, ofmap')
    parser.add_argument('--sram_banks', default=[32], nargs='*', help='number of SRAM banks. If the SRAM is not shared, give three numbers for each SRAM in order: weight, ifmap, ofmap')
    parser.add_argument('--sram_read_bandwidth', default=[32], nargs='*', help='SRAM read bandwidth. If the SRAM is not shared, give three numbers for each SRAM in order: weight, ifmap, ofmap')
    parser.add_argument('--sram_write_bandwidth', default=[32], nargs='*', help='SRAM write bandwidth. If the SRAM is not shared, give three numbers for each SRAM in order: weight, ifmap, ofmap')

    # PE array options
    parser.add_argument('--pe_x', type=int, default=14, help='The size of PE array in the x-axis')
    parser.add_argument('--pe_y', type=int, default=12, help='The size of PE array in the y-axis')
    parser.add_argument('--pe_spad_shared', default=False, action='store_true', help='If the scratchpad memory in each PE is shared among all three datatypes')
    parser.add_argument('--pe_spad_depth', default=[192, 12, 16], nargs='*', help='Scratchpad memory depth. If the scratchpad is not shared, give three numbers for each scratchpad in order: weight, ifmap, ofmap')
    parser.add_argument('--pe_spad_width', default=[16, 16, 16], nargs='*', help='Scratchpad memory width. If the scratchpad is not shared, give three numbers for each scratchpad in order: weight, ifmap, ofmap')

    # Crypt engine options
    parser.add_argument('--crypt_engine', choices=['AES-GCM'], default='AES-GCM', help='family of cryptographic engine. choices: AES-GCM')
    parser.add_argument('--crypt_engine_type', choices=['pipeline', 'parallel', 'serial'], required=True, help='type of cryptographic engine. choices: pipeline, parallel, serial')
    parser.add_argument('--crypt_engine_shared', default=False, action='store_true', help='if crypt engine is shared among all three datatypes')
    parser.add_argument('--crypt_engine_count', default=[1, 1, 1], nargs='*', help='the number of crypt engines. If not shared, give three numbers for each engine in order: weight, ifmap, ofmap')

    args = parser.parse_args()
    print(args)

    configuration_dict = {}
    configuration_dict['TEMPLATE_DESIGN'] = args.template_design
    configuration_dict['WORDBITS'] = args.word_bits
    configuration_dict['DRAM_READ_BANDWIDTH'] = args.dram_read_bandwidth
    configuration_dict['DRAM_WRITE_BANDWIDTH'] = args.dram_write_bandwidth
    configuration_dict['SRAM_SHARED'] = args.sram_shared
    configuration_dict['SRAM_DEPTH'] = [int(x) for x in args.sram_depth]
    configuration_dict['SRAM_WIDTH'] = [int(x) for x in args.sram_width]
    configuration_dict['SRAM_BANKS'] = [int(x) for x in args.sram_banks]
    configuration_dict['SRAM_READ_BANDWIDTH'] = [int(x) for x in args.sram_read_bandwidth]
    configuration_dict['SRAM_WRITE_BANDWIDTH'] = [int(x) for x in args.sram_write_bandwidth]
    configuration_dict['PE_X'] = args.pe_x
    configuration_dict['PE_Y'] = args.pe_y
    configuration_dict['PE_SPAD_SHARED'] = args.pe_spad_shared
    configuration_dict['PE_SPAD_DEPTH'] = [int(x) for x in args.pe_spad_depth]
    configuration_dict['PE_SPAD_WIDTH'] = [int(x) for x in args.pe_spad_width]

    crypt_engine_type = None
    if args.crypt_engine == 'AES-GCM':
        crypt_engine_type = 'effective_lpddr4_aesgcm'
    else:
        raise NotImplementedError()
    configuration_dict['CRYPT_ENGINE_TYPE'] = crypt_engine_type

    crypt_engine_cycle_per_block = -1
    if args.crypt_engine_type == 'pipeline':
        crypt_engine_cycle_per_block = 1
    elif args.crypt_engine_type == 'parallel':
        crypt_engine_cycle_per_block = 11
    elif args.crypt_engine_type == 'serial':
        crypt_engine_cycle_per_block = 335
    else:
        raise NotImplementedError()
    configuration_dict['CRYPT_ENGINE_CYCLE_PER_BLOCK'] = crypt_engine_cycle_per_block
    configuration_dict['CRYPT_ENGINE_SHARED'] = args.crypt_engine_shared
    configuration_dict['CRYPT_ENGINE_COUNT'] = [int(x) for x in args.crypt_engine_count]
    configuration_dict['EFFECTIVE_CONSERVATIVE'] = True
    
    # Create directory for this configuration if it doesn't exist already
    # iterate through design folders to check if any pre-exisiting folder
    design_dir = 'designs/{}'.format(configuration_dict['TEMPLATE_DESIGN'])
    arch_dir = None
    total_vers = 0
    for path in os.listdir(design_dir):
        if path != 'template' and os.path.isdir(os.path.join(design_dir, path)):
            try:
                with open(os.path.join(design_dir, path, 'config.yaml'), 'r') as f:
                    config_file = yaml.safe_load(f)
                total_vers += 1
                if config_file == configuration_dict:
                    arch_dir = path
                    print("Pre-existing folder found. Setting the arch_dir to {}".format(arch_dir))
                    break
            except:
                print("No config.yaml file in the directory {}".format(str(os.path.join(design_dir, path))))
                
    if arch_dir is None:
        arch_dir = 'ver{}'.format(total_vers)
        shutil.copytree(os.path.join(design_dir, 'template'), os.path.join(design_dir, arch_dir))
        with open(os.path.join(design_dir, arch_dir, 'config.yaml'), 'w') as f:
            _ = yaml.dump(configuration_dict, f)
        
        # create baseline and effective files
        generate_arch_files(os.path.join(design_dir, arch_dir, 'arch'), configuration_dict)
        
        # create scheduling / evaluation folder
        os.mkdir(os.path.join(design_dir, arch_dir, 'scheduling'))
        os.mkdir(os.path.join(design_dir, arch_dir, 'evaluation'))
        
        # create folders for baseline scheduling / evaluation
        os.mkdir(os.path.join(design_dir, arch_dir, 'baseline_scheduling'))
        os.mkdir(os.path.join(design_dir, arch_dir, 'baseline_evaluation'))

if __name__ == '__main__':
    main()


