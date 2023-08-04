import os
import yaml

from utils import *
from tile_analysis_utils import *
from pytorch_layer_dependency_utils import *
from authentication_block_assignment_utils import *

def get_memory_info_for_no_halo(layer_idx, layer_types, layer_tile_info_dict, u):
    result_dict = {}
    for layer_id, layer_type in zip(layer_idx, layer_types):
        tiles = layer_tile_info_dict[layer_id]['tiles']
        tile_size = layer_tile_info_dict[layer_id]['tile_size']
        n_tiles = layer_tile_info_dict[layer_id]['n_tiles']
        tiles_entire_repeat = layer_tile_info_dict[layer_id]['tiles_entire_repeat']
        
        tile_size_scalar = 1
        for s in tile_size[0]:
            tile_size_scalar *= s
        
        if u == 'tile':
            u = tile_size_scalar
        n_blocks_in_tile = math.ceil(tile_size_scalar / u)
        
        if layer_type =='weight':
            base_read = tile_size_scalar * n_tiles
            tag_read = n_blocks_in_tile * n_tiles
            result_dict[layer_id] = {'base_read': base_read, 'base_write': 0, \
                                     'tag_read': tag_read, 'tag_write': 0, \
                                     'redundant_read': 0, 'redundant_write': 0}
        elif layer_type == 'output':
            base_write = tile_size_scalar * n_tiles
            tag_write = n_blocks_in_tile * n_tiles
            base_read = tile_size_scalar * (n_tiles / tiles_entire_repeat) * (tiles_entire_repeat - 1)
            tag_read = n_blocks_in_tile * (n_tiles / tiles_entire_repeat) * (tiles_entire_repeat - 1)
            result_dict[layer_id] = {'base_read': base_read, 'base_write': base_write, \
                                     'tag_read': tag_read, 'tag_write': tag_write, \
                                     'redundant_read': 0, 'redundant_write': 0}
            
    return result_dict, u

# TODO: clean up the optional arguments for this function; some arguments are no longer used/relevant
# TODO: remove use_joint option - only used for previous version of simulated annealing with random neighbor (not from timeloop-topk)
def generate_memory_traffic_dict(n_layers, layer_info, predefined_u, predefined_perm, \
                                 base_dir, timeloop_dir, top_dir, sub_dir, \
                                 search_block_size={}, u_multiple_of=-1, WORD_SIZE=16, TAG_SIZE=64, \
                                 use_baseline=False, use_joint=False, use_joint_topk=False, evaluation_folder_given=None, \
                                 use_subset=[], prev_block_info_dict=None):
    
    memory_traffic_dict = {}
    
    # TODO: clean up use_subset and prev_block_info_dict - step 3 search time reduction, but messy now..
    search_layers_list = list(range(1, n_layers + 1)) if len(use_subset) == 0 else use_subset
    
    for layer_idx in search_layers_list:
        memory_traffic_dict[layer_idx] = {'W': {'base_read': 0, 'base_write': 0, \
                                                'tag_read': 0, 'tag_write': 0, \
                                                'redundant_read': 0, 'redundant_write': 0}, \
                                          'I': {'base_read': 0, 'base_write': 0, \
                                                'tag_read': 0, 'tag_write': 0, \
                                                'redundant_read': 0, 'redundant_write': 0}, \
                                          'O': {'base_read': 0, 'base_write': 0, \
                                                'tag_read': 0, 'tag_write': 0, \
                                                'redundant_read': 0, 'redundant_write': 0}}

    if prev_block_info_dict is None:
        block_info_dict = {}
        for layer_idx in search_layers_list:
            block_info_dict[layer_idx] = {'W': {'u': -1, 'permutation': None}, \
                                          'I': {'u': -1, 'permutation': None}, \
                                          'O': {'u': -1, 'permutation': None}}
    else:
        block_info_dict = prev_block_info_dict
    
    # 1. Weights - weights are not affected by interlayer dependency / halos
    # Assume tile as an AuthBlock strategy for weights
    for layer_id in search_layers_list:
        layer_id_for_timeloop = layer_id if use_joint_topk else layer_info[layer_id]['layer_id_for_timeloop']
        if evaluation_folder_given:
            evaluation_folder = evaluation_folder_given
        else:
            evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                            ('joint_topk' if use_joint_topk else 'evaluation'))
        xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                "timeloop-model.map+stats.xml")
        workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_id))
        layer_tile_info_dict = generate_layer_tile_info_dict([layer_id], ['weight'], [workload_file], \
                                                             [xml_file], remove_padding=False)
        # u = 512 // 16
        u = predefined_u[layer_id]['W']
        result_dict, u = get_memory_info_for_no_halo([layer_id], ['weight'], layer_tile_info_dict, u)

        memory_traffic_dict[layer_id]['W'] = result_dict[layer_id]
        block_info_dict[layer_id]['W']['u'] = u

    # 2. Inputs - consider interlayer dependency & halos
    # Note: for residual connections (or more generally when multiple layers share dependency)
    #
    visited_layers = []
    for layer_id in search_layers_list:
        # print("Processing layer {}".format(layer_id))
        if layer_id in visited_layers:
            continue

        layer_id_for_timeloop = layer_id if use_joint_topk else layer_info[layer_id]['layer_id_for_timeloop']
        prev_layer = layer_info[layer_id]['prev_layer']
        next_layer = layer_info[layer_id]['next_layer']
        dependent_prev_layer = layer_info[layer_id]['dependent_prev_layer']
        dependent_next_layer = layer_info[layer_id]['dependent_next_layer']
        
        SEARCH = search_block_size[layer_id]
        
        # Case 1: Tile-as-an-AuthBlock (no search needed)
        if predefined_u[layer_id]['I'] == 'tile' and not SEARCH:
            # print("Using tile-mac")
            # in order to support tile-size authentication block for ifmap
            # we assume that multiple tags can be associated with the same data, especially in the case of halo
            # thus, we simply assume that total read = tile size * n_tiles similar to weight and ofmap case
            if evaluation_folder_given:
                evaluation_folder = evaluation_folder_given
            else:
                evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                ('joint_topk' if use_joint_topk else 'evaluation'))
            xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                    "timeloop-model.map+stats.xml")
            workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_id))
            layer_tile_info_dict = generate_layer_tile_info_dict([layer_id], ['input'], [workload_file], \
                                                                 [xml_file], remove_padding=False)
            
            tile_size = layer_tile_info_dict[layer_id]['tile_size_before_processing']
            dimension_list = layer_tile_info_dict[layer_id]['dimension_list']
            n_tiles = layer_tile_info_dict[layer_id]['n_tiles']
            
            base_read = 0
            for size in layer_tile_info_dict[layer_id]['tile_size']:
                base_read += size[0] * size[1] * size[2] * size[3]
            dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
            tile_size_to_list = [tile_size[key] for key in dims]
            tile_size_scalar = 1
            for s in tile_size_to_list:
                tile_size_scalar *= s
            
            memory_traffic_dict[layer_id]['I']['base_read'] = base_read
            memory_traffic_dict[layer_id]['I']['redundant_read'] = tile_size_scalar * n_tiles - base_read
            memory_traffic_dict[layer_id]['I']['tag_read'] = n_tiles

            block_info_dict[layer_id]['I']['u'] = tile_size_scalar

            continue

        # Case 2: first layer of the model
        # - We don't have to consider interlayer dependency for the first layer as there is no previously computed feature map
        if len(prev_layer) == 0: # this is the first layer
            # print("this is first layer")
            if evaluation_folder_given:
                evaluation_folder = evaluation_folder_given
            else:
                evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                ('joint_topk' if use_joint_topk else 'evaluation'))
            xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                    "timeloop-model.map+stats.xml")
            workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_id))
            layer_tile_info_dict = generate_layer_tile_info_dict([layer_id], ['input'], [workload_file], \
                                                                 [xml_file], remove_padding=False)
            layer_overlap_dict, o_tiles, o_tile_size = generate_overlap_dict([layer_id], ['input'], \
                                                                             layer_tile_info_dict, reference_layer_idx=layer_id)
            dims_info, dims_score, permutation = classify_dims(layer_overlap_dict, o_tiles, o_tile_size)

            if SEARCH:
                multilayer_result_dict, min_u = search_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                        o_tiles, o_tile_size, \
                                                                        dims_info, dims_score, permutation, \
                                                                        word_size=WORD_SIZE, tag_size=TAG_SIZE, u_multiple_of=u_multiple_of)
            else:
                u = predefined_u[layer_id]['I']
                permutation = predefined_perm[layer_id]['I']
                multilayer_result_dict = evaluate_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                   o_tiles, o_tile_size, \
                                                                   dims_info, dims_score, permutation, u, \
                                                                   word_size=WORD_SIZE, tag_size=TAG_SIZE)
                min_u = u
            memory_traffic_dict[layer_id]['I']['base_read'] = multilayer_result_dict[layer_id]['base_read']
            memory_traffic_dict[layer_id]['I']['redundant_read'] = multilayer_result_dict[layer_id]['redundant_read']
            memory_traffic_dict[layer_id]['I']['tag_read'] = multilayer_result_dict[layer_id]['tag_read']

            block_info_dict[layer_id]['I']['u'] = min_u
            block_info_dict[layer_id]['I']['permutation'] = permutation

            visited_layers.append(layer_id)

        # Case 3: Not the first layer, but no dependency with the previous layer
        elif len(dependent_prev_layer) == 0:
            # print("not a first layer but no dependent prev layer")
            # we have to find 'shared' layers
            shared_layer = [layer_id]
            
            # find other layers that have exactly same prev layer (e.g., residual branches)
            # if two or more layers share the exactly same prev layer - they use the same input feature map
            # find a unified auth block assignment considering those layers
            for key in layer_info.keys():
                if key != layer_id and layer_info[key]['prev_layer'] == prev_layer:
                    shared_layer.append(key)
            xml_file_list = []
            workload_file_list = []
            layer_idx_list = []
            layer_type_list = []
            for i in shared_layer:
                layer_id_for_timeloop = i if use_joint_topk else layer_info[i]['layer_id_for_timeloop']
                if evaluation_folder_given:
                    evaluation_folder = evaluation_folder_given
                else:
                    evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                    ('joint_topk' if use_joint_topk else 'evaluation'))
                xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                        "timeloop-model.map+stats.xml")
                workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, i))
                xml_file_list.append(xml_file)
                workload_file_list.append(workload_file)
                layer_idx_list.append(i)
                layer_type_list.append('input')
            layer_tile_info_dict = generate_layer_tile_info_dict(layer_idx_list, layer_type_list, \
                                                                 workload_file_list, xml_file_list, remove_padding=True)

            # TODO: clean up
            # for reference layers, their tiles_for_reference should **cover** the ifmap
            # problematic cases: large strides and only part of ifmap is used
            # check cover: is there quick and accurate method of check if a list of subsets covers our set of interest?
            # for now, implement the most naive possible version...
            if len(shared_layer) > 1:
                reference_idx_candidates = []
                for i in layer_tile_info_dict.keys():
                    tiles = layer_tile_info_dict[i]['tiles']
                    tile_size = layer_tile_info_dict[i]['tile_size']
                    ifmap_size = layer_tile_info_dict[i]['ifmap_size']
                    # print(ifmap_size)
                    full_index_list = []
                    for n in range(ifmap_size[0]):
                        for c in range(ifmap_size[1]):
                            for h in range(ifmap_size[2]):
                                for w in range(ifmap_size[3]):
                                    full_index_list.append((n, c, h, w))
                    subset_index_list = []
                    for tile, size in zip(tiles, tile_size):
                        temp_list = []
                        for n in range(size[0]):
                            for c in range(size[1]):
                                for h in range(size[2]):
                                    for w in range(size[3]):
                                        temp_list.append((tile[0] + n, tile[1] + c, tile[2] + h, tile[3] + w))
                        subset_index_list = list(set(subset_index_list).union(set(temp_list)))
                    # print(set(full_index_list) == set(subset_index_list))
                    if set(full_index_list) == set(subset_index_list):
                        reference_idx_candidates.append(i)
            else:
                reference_idx_candidates = [layer_id]
            # print(reference_idx_candidates)

            # if multiple reference idx exist, then we should choose the most optimal one
            overhead_per_reference_idx = {}
            result_dict_per_reference_idx = {}
            for reference_idx in reference_idx_candidates:
                layer_overlap_dict, o_tiles, o_tile_size = generate_overlap_dict(layer_idx_list, layer_type_list, \
                                                                                 layer_tile_info_dict, reference_layer_idx=reference_idx)
                dims_info, dims_score, permutation = classify_dims(layer_overlap_dict, o_tiles, o_tile_size)
                if SEARCH:
                    multilayer_result_dict, min_u = search_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                            o_tiles, o_tile_size, \
                                                                            dims_info, dims_score, permutation, \
                                                                            word_size=WORD_SIZE, tag_size=TAG_SIZE, u_multiple_of=u_multiple_of)
                else:
                    u = predefined_u[layer_id]['I']
                    permutation = predefined_perm[layer_id]['I']
                    multilayer_result_dict = evaluate_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                       o_tiles, o_tile_size, \
                                                                       dims_info, dims_score, permutation, u, \
                                                                       word_size=WORD_SIZE, tag_size=TAG_SIZE)
                    min_u = u
                overhead = 0
                for i in multilayer_result_dict.keys():
                    overhead += (multilayer_result_dict[i]['base_read'] + multilayer_result_dict[i]['redundant_read']) * WORD_SIZE \
                                + multilayer_result_dict[i]['tag_read'] * TAG_SIZE
                overhead_per_reference_idx[reference_idx] = overhead
                result_dict_per_reference_idx[reference_idx] = {'multilayer_result_dict': multilayer_result_dict, \
                                                                'min_u': min_u, 'permutation': permutation}

            min_overhead_reference_idx = min(overhead_per_reference_idx, key=overhead_per_reference_idx.get)
            multilayer_result_dict = result_dict_per_reference_idx[min_overhead_reference_idx]['multilayer_result_dict']
            min_u = result_dict_per_reference_idx[min_overhead_reference_idx]['min_u']
            permutation = result_dict_per_reference_idx[min_overhead_reference_idx]['permutation']

            # print(multilayer_result_dict.keys())
            for i in multilayer_result_dict.keys():
                memory_traffic_dict[i]['I']['base_read'] = multilayer_result_dict[i]['base_read']
                memory_traffic_dict[i]['I']['redundant_read'] = multilayer_result_dict[i]['redundant_read']
                memory_traffic_dict[i]['I']['tag_read'] = multilayer_result_dict[i]['tag_read']

                block_info_dict[i]['I']['u'] = min_u
                block_info_dict[i]['I']['permutation'] = permutation
                block_info_dict[i]['I']['shared'] = shared_layer
                block_info_dict[i]['I']['reference_layer'] = min_overhead_reference_idx

                visited_layers.append(i)
                
        # Case 4: Interlayer dependency has to be considered
        elif len(dependent_prev_layer) > 0:
            # print("there is a dependent prev layer")
            # we have to find 'shared' layers
            shared_layer = [layer_id]
            # find other layers that have exactly same prev layer
            for key in layer_info.keys():
                if key != layer_id and layer_info[key]['dependent_prev_layer'] == dependent_prev_layer:
                    shared_layer.append(key)
            xml_file_list = []
            workload_file_list = []
            layer_idx_list = []
            layer_type_list = []
            for i in shared_layer:
                layer_id_for_timeloop = i if use_joint_topk else layer_info[i]['layer_id_for_timeloop']
                if evaluation_folder_given:
                    evaluation_folder = evaluation_folder_given
                else:
                    evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                    ('joint_topk' if use_joint_topk else 'evaluation'))
                xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                        "timeloop-model.map+stats.xml")
                workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, i))
                xml_file_list.append(xml_file)
                workload_file_list.append(workload_file)
                layer_idx_list.append(i)
                layer_type_list.append('input')

            # for now, we will consider len(dependent_prev_layer) == 1
            # TODO: multiple dependent prev layers
            for i in dependent_prev_layer:
                layer_id_for_timeloop = i if use_joint_topk else layer_info[i]['layer_id_for_timeloop']
                if evaluation_folder_given:
                    evaluation_folder = evaluation_folder_given
                else:
                    evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                    ('joint_topk' if use_joint_topk else 'evaluation'))
                xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                        "timeloop-model.map+stats.xml")
                workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, i))
                xml_file_list.append(xml_file)
                workload_file_list.append(workload_file)
                layer_idx_list.append(i)
                layer_type_list.append('output')
            # print(xml_file_list)

            layer_tile_info_dict = generate_layer_tile_info_dict(layer_idx_list, layer_type_list, \
                                                                 workload_file_list, xml_file_list, remove_padding=True)

            layer_overlap_dict, o_tiles, o_tile_size = generate_overlap_dict(layer_idx_list, layer_type_list, \
                                                                             layer_tile_info_dict, reference_layer_idx=-1)
            dims_info, dims_score, permutation = classify_dims(layer_overlap_dict, o_tiles, o_tile_size)
            if SEARCH:
                multilayer_result_dict, min_u = search_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                        o_tiles, o_tile_size, \
                                                                        dims_info, dims_score, permutation, \
                                                                        word_size=WORD_SIZE, tag_size=TAG_SIZE, u_multiple_of=u_multiple_of)
            else:
                u = predefined_u[layer_id]['I']
                permutation = predefined_perm[layer_id]['I']
                multilayer_result_dict = evaluate_block_multilayer(layer_tile_info_dict, layer_overlap_dict, \
                                                                   o_tiles, o_tile_size, \
                                                                   dims_info, dims_score, permutation, u, \
                                                                   word_size=WORD_SIZE, tag_size=TAG_SIZE)
                min_u = u

            for i in multilayer_result_dict.keys():
                memory_traffic_dict[i]['I']['base_read'] = multilayer_result_dict[i]['base_read']
                memory_traffic_dict[i]['I']['redundant_read'] = multilayer_result_dict[i]['redundant_read']
                memory_traffic_dict[i]['I']['tag_read'] = multilayer_result_dict[i]['tag_read']

                block_info_dict[i]['I']['u'] = min_u
                block_info_dict[i]['I']['permutation'] = permutation
                block_info_dict[i]['I']['shared'] = shared_layer

                visited_layers.append(i)

    # 3. Output - if it was considered for interlayer dependency when considering inputs, 
    #             AuthBlock assignment identified for inputs should be used
    #             Otherwise, if there is no interlayer dependency for one ofmap, we can use tile-as-an-AuthBlock
    for layer_id in search_layers_list:
        layer_id_for_timeloop = layer_id if use_joint_topk else layer_info[layer_id]['layer_id_for_timeloop']
        dependent_next_layer = layer_info[layer_id]['dependent_next_layer']
        if evaluation_folder_given:
            evaluation_folder = evaluation_folder_given
        else:
            evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                            ('joint_topk' if use_joint_topk else 'evaluation'))
        xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                "timeloop-model.map+stats.xml")
        workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_id))
        layer_tile_info_dict = generate_layer_tile_info_dict([layer_id], ['output'], [workload_file], \
                                                             [xml_file], remove_padding=False)

        # if no dependency
        if len(dependent_next_layer) == 0:
            u = predefined_u[layer_id]['O']
            result_dict, u = get_memory_info_for_no_halo([layer_id], ['output'], layer_tile_info_dict, u)

            memory_traffic_dict[layer_id]['O'] = result_dict[layer_id]
            block_info_dict[layer_id]['O']['u'] = u

        else:
            u = block_info_dict[dependent_next_layer[0]]['I']['u']
            result_dict, u = get_memory_info_for_no_halo([layer_id], ['output'], layer_tile_info_dict, u)

            memory_traffic_dict[layer_id]['O'] = result_dict[layer_id]
            block_info_dict[layer_id]['O']['u'] = u
            
    return memory_traffic_dict, block_info_dict

def generate_rehash_info_dict(n_layers, layer_info, block_info_dict, \
                              base_dir, timeloop_dir, top_dir, sub_dir, use_baseline=False, use_joint=False, use_joint_topk=False, \
                              evaluation_folder_given=None, use_subset=[]):
    rehash_info_dict = {}
    search_layers_list = list(range(1, n_layers + 1)) if len(use_subset) == 0 else use_subset
    for layer_id in search_layers_list:
        layer_id_for_timeloop = layer_id if use_joint_topk else layer_info[layer_id]['layer_id_for_timeloop']
        prev_layer = layer_info[layer_id]['prev_layer']
        next_layer = layer_info[layer_id]['next_layer']
        dependent_prev_layer = layer_info[layer_id]['dependent_prev_layer']
        dependent_next_layer = layer_info[layer_id]['dependent_next_layer']

        rehash_layers = [x for x in next_layer if x not in dependent_next_layer]

        for rehash_layer in rehash_layers:
            # record memory traffic required for rehashing
            # - base read of encrypted data + tag read
            # - base write of decrypted data + tag write

            # ofmap - read the encrypted data + read tags
            if evaluation_folder_given:
                evaluation_folder = evaluation_folder_given
            else:
                evaluation_folder = 'baseline_evaluation' if use_baseline else ('joint' if use_joint else \
                                                                                ('joint_topk' if use_joint_topk else 'evaluation'))
            xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}".format(layer_id_for_timeloop), \
                                    "timeloop-model.map+stats.xml")
            workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, layer_id))
            layer_tile_info_dict = generate_layer_tile_info_dict([layer_id], ['output'], [workload_file], \
                                                                 [xml_file], remove_padding=False)

            tiles = layer_tile_info_dict[layer_id]['tiles']
            tile_size = layer_tile_info_dict[layer_id]['tile_size']
            n_tiles = layer_tile_info_dict[layer_id]['n_tiles']
            tiles_entire_repeat = layer_tile_info_dict[layer_id]['tiles_entire_repeat']

            tile_size_scalar = 1
            for s in tile_size[0]:
                tile_size_scalar *= s

            u = block_info_dict[layer_id]['O']['u']
            n_blocks_in_tile = math.ceil(tile_size_scalar / u)

            base_read = tile_size_scalar * n_tiles / tiles_entire_repeat # we only have to read once!
            tag_read = n_blocks_in_tile * n_tiles / tiles_entire_repeat

            # ifmap - write the re-encrypted data + write tags
            o_layer_id_for_timeloop = rehash_layer if use_joint_topk else layer_info[rehash_layer]['layer_id_for_timeloop']
            xml_file = os.path.join(base_dir, timeloop_dir, evaluation_folder, sub_dir, "layer{}"\
                                    .format(o_layer_id_for_timeloop),
                                    "timeloop-model.map+stats.xml")
            workload_file = os.path.join(base_dir, top_dir, sub_dir, "{}_layer{}.yaml".format(sub_dir, rehash_layer))
            layer_tile_info_dict = generate_layer_tile_info_dict([rehash_layer], ['input'], [workload_file], \
                                                                 [xml_file], remove_padding=True)
            layer_overlap_dict, o_tiles, o_tile_size = generate_overlap_dict([rehash_layer], ['input'], \
                                                                             layer_tile_info_dict, reference_layer_idx=rehash_layer)
            # dims_info, dims_score, permutation = classify_dims(layer_overlap_dict, o_tiles, o_tile_size)

            tile_size_before_processing = layer_tile_info_dict[rehash_layer]['tile_size_before_processing']
            dimension_list = layer_tile_info_dict[rehash_layer]['dimension_list']
            n_tiles = layer_tile_info_dict[rehash_layer]['n_tiles']
            tiles_entire_repeat = layer_tile_info_dict[rehash_layer]['tiles_entire_repeat']
            dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
            tile_size_to_list = [tile_size_before_processing[key] for key in dims]
            tile_size_scalar = 1
            for s in tile_size_to_list:
                tile_size_scalar *= s
            
            u = block_info_dict[rehash_layer]['I']['u']
            if u == tile_size_scalar:
                base_write = tile_size_scalar * n_tiles / tiles_entire_repeat
                tag_write = n_tiles / tiles_entire_repeat
            else:
                base_write = 0
                tag_write = 0
                for o_idx, o_tile in enumerate(o_tiles):
                    curr_tile_size = 1
                    for s in o_tile_size[o_idx]:
                        curr_tile_size *= s
                    base_write += curr_tile_size
                    tag_write += math.ceil(curr_tile_size / u)

            rehash_info_dict[(layer_id, rehash_layer)] = {'base_read': base_read, 'base_write': base_write, \
                                                          'tag_read': tag_read, 'tag_write': tag_write, \
                                                          'redundant_read': 0, 'redundant_write': 0}
    return rehash_info_dict

def convert_traffic_to_action(base_read, base_write, redundant_read, redundant_write, tag_read, tag_write, \
                              mac_block_size, aes_block_size, word_size, tag_size):
    aes_engine_count = tag_read * (math.ceil(mac_block_size / aes_block_size) + 1) + \
                       tag_write * (math.ceil(mac_block_size / aes_block_size) + 1)
    gf_mult_count = tag_read * math.ceil(mac_block_size / aes_block_size) + tag_write * math.ceil(mac_block_size / aes_block_size)
    xor_count = 2 * (tag_read * math.ceil(mac_block_size / aes_block_size) + tag_write * math.ceil(mac_block_size / aes_block_size))
    
    total_read_bits = (base_read + redundant_read) * word_size + tag_read * tag_size
    total_write_bits = base_write * word_size + tag_write * tag_size
    
    additional_read_bits = redundant_read * word_size + tag_read * tag_size
    additional_write_bits = tag_write * tag_size
    
    tag_bits = (tag_read * tag_size) + (tag_write * tag_size)
    redundant_bits = redundant_read * word_size
    
    return aes_engine_count, gf_mult_count, xor_count, \
            total_read_bits, total_write_bits, additional_read_bits, additional_write_bits, \
            tag_bits, redundant_bits

def get_action_dict(n_layers, memory_traffic_dict, block_info_dict, \
                    WORD_SIZE, TAG_SIZE, AES_DATAPATH=128):
    cryptographic_action_count_dict = {}
    for layer_id in memory_traffic_dict.keys():
        cryptographic_action_count_dict[layer_id] = {}
        
        # Weights
        base_read = memory_traffic_dict[layer_id]['W']['base_read']
        base_write = memory_traffic_dict[layer_id]['W']['base_write']
        redundant_read = memory_traffic_dict[layer_id]['W']['redundant_read']
        redundant_write = memory_traffic_dict[layer_id]['W']['redundant_write']
        tag_read = memory_traffic_dict[layer_id]['W']['tag_read']
        tag_write = memory_traffic_dict[layer_id]['W']['tag_write']

        mac_block_size = block_info_dict[layer_id]['W']['u'] * WORD_SIZE

        aes_engine_count, gf_mult_count, xor_count, total_read_bits, total_write_bits, \
        additional_read_bits, additional_write_bits, tag_bits, redundant_bits = \
        convert_traffic_to_action(base_read, base_write, redundant_read, redundant_write, tag_read, tag_write, \
                                  mac_block_size, AES_DATAPATH, WORD_SIZE, TAG_SIZE)
        cryptographic_action_count_dict[layer_id]['W'] = {'aes_engine_count': aes_engine_count, \
                                                         'gf_mult_count': gf_mult_count, \
                                                         'xor_count': xor_count, \
                                                         'total_read_bits': total_read_bits, \
                                                         'total_write_bits': total_write_bits, \
                                                         'additional_read_bits': additional_read_bits, \
                                                         'additional_write_bits': additional_write_bits, \
                                                         'tag_bits': tag_bits, \
                                                         'redundant_bits': redundant_bits}

        # Inputs
        base_read = memory_traffic_dict[layer_id]['I']['base_read']
        base_write = memory_traffic_dict[layer_id]['I']['base_write']
        redundant_read = memory_traffic_dict[layer_id]['I']['redundant_read']
        redundant_write = memory_traffic_dict[layer_id]['I']['redundant_write']
        tag_read = memory_traffic_dict[layer_id]['I']['tag_read']
        tag_write = memory_traffic_dict[layer_id]['I']['tag_write']

        mac_block_size = block_info_dict[layer_id]['I']['u'] * WORD_SIZE

        aes_engine_count, gf_mult_count, xor_count, total_read_bits, total_write_bits, \
        additional_read_bits, additional_write_bits, tag_bits, redundant_bits = \
        convert_traffic_to_action(base_read, base_write, redundant_read, redundant_write, tag_read, tag_write, \
                                  mac_block_size, AES_DATAPATH, WORD_SIZE, TAG_SIZE)
        cryptographic_action_count_dict[layer_id]['I'] = {'aes_engine_count': aes_engine_count, \
                                                         'gf_mult_count': gf_mult_count, \
                                                         'xor_count': xor_count, \
                                                         'total_read_bits': total_read_bits, \
                                                         'total_write_bits': total_write_bits, \
                                                         'additional_read_bits': additional_read_bits, \
                                                         'additional_write_bits': additional_write_bits, \
                                                         'tag_bits': tag_bits, \
                                                         'redundant_bits': redundant_bits}

        # Outputs
        base_read = memory_traffic_dict[layer_id]['O']['base_read']
        base_write = memory_traffic_dict[layer_id]['O']['base_write']
        redundant_read = memory_traffic_dict[layer_id]['O']['redundant_read']
        redundant_write = memory_traffic_dict[layer_id]['O']['redundant_write']
        tag_read = memory_traffic_dict[layer_id]['O']['tag_read']
        tag_write = memory_traffic_dict[layer_id]['O']['tag_write']

        mac_block_size = block_info_dict[layer_id]['O']['u'] * WORD_SIZE

        aes_engine_count, gf_mult_count, xor_count, total_read_bits, total_write_bits, \
        additional_read_bits, additional_write_bits, tag_bits, redundant_bits = \
        convert_traffic_to_action(base_read, base_write, redundant_read, redundant_write, tag_read, tag_write, \
                                  mac_block_size, AES_DATAPATH, WORD_SIZE, TAG_SIZE)
        cryptographic_action_count_dict[layer_id]['O'] = {'aes_engine_count': aes_engine_count, \
                                                         'gf_mult_count': gf_mult_count, \
                                                         'xor_count': xor_count, \
                                                         'total_read_bits': total_read_bits, \
                                                         'total_write_bits': total_write_bits, \
                                                         'additional_read_bits': additional_read_bits, \
                                                         'additional_write_bits': additional_write_bits, \
                                                         'tag_bits': tag_bits, \
                                                         'redundant_bits': redundant_bits}
    return cryptographic_action_count_dict

def get_action_dict_for_rehash(rehash_info_dict, block_info_dict, WORD_SIZE, TAG_SIZE, AES_DATAPATH=128):
    rehash_action_count_dict = {}
    branch_out_layers = []
    branch_in_layers = []
    for key in rehash_info_dict.keys():
        out_layer = key[0]
        in_layer = key[1]
        
        base_read = rehash_info_dict[key]['base_read']
        tag_read = rehash_info_dict[key]['tag_read']
        base_write = rehash_info_dict[key]['base_write']
        tag_write = rehash_info_dict[key]['tag_write']
        
        # in case of branching out (one ofmap serving multiple future ifmaps)
        # we only have to read encrypted ofmap of this layer once - then rehash for several other ifmaps
        if out_layer in branch_out_layers:
            base_read = 0
            tag_read = 0
        else:
            branch_out_layers.append(out_layer)
        # for branching in, similar argument can be made (multiple ofmaps serving one future ifmap)
        # and we only consider one rehashing (read necessary ofmaps and add them, and rehash once)
        if in_layer in branch_in_layers:
            base_write = 0
            tag_write = 0
        else:
            branch_in_layers.append(in_layer)
            
        # aes_engine_count, gf_mult_count, xor_count, total_read_bits, total_write_bits, additional_read_bits, additional_write_bits = \
        # convert_traffic_to_action(base_read, base_write, 0, 0, tag_read, tag_write, \
        #                           mac_block_size, AES_DATAPATH, WORD_SIZE, TAG_SIZE)
        aes_engine_count = tag_read * (WORD_SIZE * block_info_dict[out_layer]['O']['u'] / AES_DATAPATH + 1) + \
                           tag_write * (WORD_SIZE * block_info_dict[in_layer]['I']['u'] / AES_DATAPATH + 1)
        gf_mult_count = tag_read * (WORD_SIZE * block_info_dict[out_layer]['O']['u'] / AES_DATAPATH) + \
                        tag_write * (WORD_SIZE * block_info_dict[in_layer]['I']['u'] / AES_DATAPATH)
        xor_count = 2 * (tag_read * (WORD_SIZE * block_info_dict[out_layer]['O']['u'] / AES_DATAPATH) + \
                         tag_write * (WORD_SIZE * block_info_dict[in_layer]['I']['u'] / AES_DATAPATH))

        total_read_bits = base_read * WORD_SIZE + tag_read * TAG_SIZE
        total_write_bits = base_write * WORD_SIZE + tag_write * TAG_SIZE

        additional_read_bits = tag_read * TAG_SIZE
        additional_write_bits = tag_write * TAG_SIZE
    
        
        rehash_action_count_dict[key] = {'aes_engine_count': aes_engine_count, \
                                         'gf_mult_count': gf_mult_count, \
                                         'xor_count': xor_count, \
                                         'total_read_bits': total_read_bits, \
                                         'total_write_bits': total_write_bits, \
                                         'additional_read_bits': additional_read_bits, \
                                         'additional_write_bits': additional_write_bits}
        
    return rehash_action_count_dict