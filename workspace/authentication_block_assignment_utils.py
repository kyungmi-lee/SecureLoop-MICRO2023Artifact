import numpy as np
import copy
import math
import xml.etree.ElementTree as ET
import os
import yaml
from collections import OrderedDict
from collections import Counter
from itertools import islice

from tile_analysis_utils import *

def is_overlap_multilayer(tile1, tile_size_1, tile2, tile_size_2, n_dim=4):
    overlap = True
    overlap_start_idx = [0] * n_dim
    overlap_end_idx = [0] * n_dim
    for d in range(n_dim):
        # case 1: i_tile larger than o_tile in this dimension, and o_tile compleltely lies within i_tile
        if tile1[d] < tile2[d] and tile1[d] + tile_size_1[d] > tile2[d] + tile_size_2[d]:
            overlap_start_idx[d] = tile2[d]
            overlap_end_idx[d] = tile2[d] + tile_size_2[d]
        # case 2: they overlap at the left edge of i_tile
        elif tile2[d] <= tile1[d] and tile1[d] < tile2[d] + tile_size_2[d] and tile2[d] + tile_size_2[d] < tile1[d] + tile_size_1[d]:
            overlap_start_idx[d] = tile1[d]
            overlap_end_idx[d] = tile2[d] + tile_size_2[d]
        # case 3: they overlap at the right edge of i_tile
        elif tile1[d] < tile2[d] and tile2[d] < tile1[d] + tile_size_1[d] and tile1[d] + tile_size_1[d] <= tile2[d] + tile_size_2[d]:
            overlap_start_idx[d] = tile2[d]
            overlap_end_idx[d] = tile1[d] + tile_size_1[d]
        # case 4: o_tile larger than i_tile in this dimension, and i_tile completely lies within o_tile
        elif tile2[d] <= tile1[d] and tile1[d] + tile_size_1[d] <= tile2[d] + tile_size_2[d]:
            overlap_start_idx[d] = tile1[d]
            overlap_end_idx[d] = tile1[d] + tile_size_1[d]
        else:
            overlap = False
    return overlap, overlap_start_idx, overlap_end_idx

def process_input_feature_map_multilayer(tiles, halos, tile_size, padding_h, padding_w, size_h, size_w, dimension_list, \
                                         remove_padding=True):
    dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    tile_size_to_list = [tile_size[key] for key in dims]
    processed_tiles = []
    processed_tile_size = []
    for tile_idx in range(len(tiles)):
        new_tile_start_idx = copy.deepcopy(tiles[tile_idx])
        new_tile_size = copy.deepcopy(tile_size_to_list)
        # record the starting idx in each dimension
        # first, check if there's halo with the previous thile
        for h in halos:
            if h[1] == tile_idx and h[0] == tile_idx - 1: # consecutive tile
                halo_dim = h[4][0]
                new_tile_start_idx[halo_dim] = h[3][halo_dim]
                new_tile_size[halo_dim] -= h[3][halo_dim] - h[2][halo_dim]
                
        # second, check if there's padding
        # min_h = padding_h, max_h = size_h - padding_h ...
        if remove_padding:
            clip_h = False
            clip_w = False

            if new_tile_start_idx[2] < padding_h:
                clip_h = True
            if new_tile_start_idx[3] < padding_w:
                clip_w = True
            if new_tile_start_idx[2] + new_tile_size[2] > size_h - padding_h:
                new_tile_size[2] -= (new_tile_start_idx[2] + new_tile_size[2] - (size_h - padding_h))
            if new_tile_start_idx[3] + new_tile_size[3] > size_w - padding_w:
                new_tile_size[3] -= (new_tile_start_idx[3] + new_tile_size[3] - (size_w - padding_w))

            if clip_h: 
                new_tile_size[2] -= (padding_h - new_tile_start_idx[2])
                new_tile_start_idx[2] = padding_h

            if clip_w:
                new_tile_size[3] -= (padding_w - new_tile_start_idx[3])
                new_tile_start_idx[3] = padding_w
        
        processed_tiles.append(new_tile_start_idx)
        for d_idx in range(len(new_tile_size)):
            if new_tile_size[d_idx] < 0:
                new_tile_size[d_idx] = 0
        processed_tile_size.append(new_tile_size)
            
    if remove_padding:
        for t in processed_tiles:
            t[2] -= padding_h
            t[3] -= padding_w

    # find halos
    halos_processed = []
    duplicates = {} # internal book-keeping for duplicated tiles
    for i in range(len(processed_tiles)):
        duplicates[i] = [i]
    for idx1 in range(len(processed_tiles) - 1):
        for idx2 in range(idx1 + 1, len(processed_tiles)):
            # check overlap in each dimension
            size_1, size_2 = 1, 1
            for s in processed_tile_size[idx1]:
                size_1 *= s
            for s in processed_tile_size[idx2]:
                size_2 *= s
            if size_1 == 0 or size_2 == 0:
                continue
            overlap, overlap_start_idx, overlap_end_idx = is_overlap_multilayer(processed_tiles[idx1], processed_tile_size[idx1], \
                                                                                processed_tiles[idx2], processed_tile_size[idx2], \
                                                                                len(dims))
            duplicate = is_duplicate(processed_tiles[idx1], processed_tiles[idx2])
            if duplicate:
                duplicates[idx1].append(idx2)
                duplicates[idx2].append(idx1)
            elif overlap:
                # identify in which dimension halo exists
                halo_dim = []
                for i, (s, e) in enumerate(zip(overlap_start_idx, overlap_end_idx)):
                    if e - s != processed_tile_size[idx2][i]:
                        halo_dim.append(i)
                # check if idx1 has any duplicates: if so, only the last duplicated tile should be considered
                duplicate_idx1 = duplicates[idx1]
                if idx1 == max(duplicate_idx1):
                    halo_info = (idx1, idx2, overlap_start_idx, overlap_end_idx, halo_dim)
                    halos_processed.append(halo_info)
    
    return processed_tiles, processed_tile_size, halos_processed

# First, we should get tiles and halos information for all needed inputs
# we have to store information for each layer
# - n_tiles, tiles_processed, tile_size_processed, halos_processed, tiles_for_tag_count
def generate_layer_tile_info_dict(layer_idx, layer_types, workload_paths, xml_paths, remove_padding):
    layer_tile_info_dict = {}
    for layer_id, layer_type, workload_path, xml_path in zip(layer_idx, layer_types, workload_paths, xml_paths):
        # get loopnest and workload info
        loopnest, mapping_info = parse_xml_file(xml_path)
        dimension_list, (stride_h, stride_w), (kernel_size_r, kernel_size_s), \
        (padding_h, padding_w), (N, C, P, Q), dw = parse_workload_file(workload_path)
        tile_size, num_tiles, tile_idx_info, tiles_entire_repeat = get_tiling_info(loopnest, mapping_info, dimension_list, \
                                                                                   kernel_size_r, kernel_size_s, stride_h, stride_w, \
                                                                                   padding_h, padding_w, layer_type, dw)
        if layer_type == 'input':
            dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
        elif layer_type == 'output':
            dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'] if dw else ['N', 'M', 'P', 'Q'])
        elif layer_type == 'weight':
            dims = get_dimension_idx(dimension_list, ['C', 'R', 'S'] if dw else ['M', 'C', 'R', 'S'])
        
        n_tiles, tiles, halo = find_tiles_and_halos(tile_size, num_tiles, tiles_entire_repeat, \
                                                    tile_idx_info, dimension_list, datatype=layer_type, \
                                                    check_halo=(layer_type=='input'), dw=dw)

        # remove halos from consecutive tiles (from above), then return "processed" tile info and halos
        # - once the halos are removed, each tile can have different size: so return both tile info and size
        # - also, identify remaining halos between non-consecutive tiles
        # - these halos have to be loaded/stored from/to off-chip since they arise from sliding window patterns
        # tiles_processed, tile_size_processed, halos_processed = remove_halos_from_consecutive_tiles(copy.deepcopy(tiles), halo, \
        #                                                                                             copy.deepcopy(tile_size), dimension_list)
        if layer_type == 'input':
            size_h = (P - 1) * stride_h + kernel_size_r
            size_w = (Q - 1) * stride_w + kernel_size_s
            tiles_processed, tile_size_processed, halos_processed = process_input_feature_map_multilayer(tiles, halo, \
                                                                                                         tile_size, padding_h, padding_w, \
                                                                                                         size_h, size_w, dimension_list, \
                                                                                                         remove_padding)
            # print(halos_processed)
            # finally, we have to convert halo information such that we can identify 'unique' patterns
            # also, we will return tiles_for_tag_count, which will be useful when computing the total tag counts assuming only one tag per one data
            tiles_for_tag_count, unique_halo, unique_halo_counts = process_halos(copy.deepcopy(tiles_processed), halos_processed)

            info_dict = {}
            info_dict['tiles'] = tiles_processed
            info_dict['tile_size'] = tile_size_processed
            info_dict['tiles_for_reference'] = tiles_for_tag_count
            info_dict['dimension_list'] = dimension_list
            info_dict['ifmap_size'] = [N, C, size_h - 2 * padding_h, size_w - 2 * padding_w] if remove_padding else  [N, C, size_h, size_w]
            info_dict['tiles_entire_repeat'] = tiles_entire_repeat
            info_dict['tile_size_before_processing'] = tile_size
            info_dict['n_tiles'] = n_tiles

        elif layer_type == 'output' or layer_type == 'weight':
            info_dict = {}
            info_dict['tiles'] = tiles
            tile_size_list = [tile_size[key] for key in dims]
            info_dict['tile_size'] = [tile_size_list] * len(tiles)
            info_dict['tiles_for_reference'] = tiles
            info_dict['dimension_list'] = dimension_list
            info_dict['n_tiles'] = n_tiles
            info_dict['tiles_entire_repeat'] = tiles_entire_repeat

        layer_tile_info_dict[layer_id] = info_dict
        
    return layer_tile_info_dict

# Next, run the search N times
# - each time, we set one layer to be 'reference' - so that its tiles_for_tag_count will be used as a constraint for assinging blocks
# - for all layers, we should find 'overlaps' with this reference macro-tiles, and run the inter-layer analysis
def find_overlaps_multilayer(o_tiles, o_tile_size_processed, i_tiles_processed, i_tile_size_processed):
    i_tile_overlap_dict = {}
    for idx in range(len(i_tiles_processed)):
        i_tile_overlap_dict[idx] = []
    # format for storing the overlapping tiles:
    # i_tile_overlap_dict[idx_of_input_tile] = [(idx_of_output_tile, overlap_start_idx, overlap_end_idx), (), ...]
    for i_idx, tile in enumerate(i_tiles_processed):
        tile_size = i_tile_size_processed[i_idx]
        for o_idx, o_tile in enumerate(o_tiles):
            size_1, size_2 = 1, 1
            for s in tile_size:
                size_1 *= s
            for s in o_tile_size_processed[o_idx]:
                size_2 *= s
            if size_1 == 0 or size_2 == 0:
                continue
            overlap, overlap_start_idx, overlap_end_idx = is_overlap_multilayer(tile, tile_size, o_tile, o_tile_size_processed[o_idx])
            if overlap:
                i_tile_overlap_dict[i_idx].append((o_idx, overlap_start_idx, overlap_end_idx))
    
    return i_tile_overlap_dict

def generate_overlap_dict(layer_idx, layer_types, layer_tile_info_dict, reference_layer_idx=-1):
    if reference_layer_idx == -1:
        reference_layer_idx = layer_idx[layer_types.index('output')]
        o_tiles = layer_tile_info_dict[reference_layer_idx]['tiles_for_reference']
        o_tile_size = layer_tile_info_dict[reference_layer_idx]['tile_size']
        o_dimension_list = layer_tile_info_dict[reference_layer_idx]['dimension_list']
    else:
        o_tiles = layer_tile_info_dict[reference_layer_idx]['tiles_for_reference']
        o_tile_base = layer_tile_info_dict[reference_layer_idx]['tiles']
        o_tile_base_size = layer_tile_info_dict[reference_layer_idx]['tile_size']
        o_tile_size = []
        for tile_idx in range(len(o_tiles)):
            curr_tile_size = []
            for (t, ts, tt) in zip(o_tile_base[tile_idx], o_tile_base_size[tile_idx], o_tiles[tile_idx]):
                curr_tile_size.append(ts - (tt - t))
            o_tile_size.append(curr_tile_size)
        o_dimension_list = layer_tile_info_dict[reference_layer_idx]['dimension_list']
    
    layer_overlap_dict = {}
    for layer_id, layer_type in zip(layer_idx, layer_types):
        if layer_type == 'output':
            continue
        i_tiles = layer_tile_info_dict[layer_id]['tiles']
        i_tile_size = layer_tile_info_dict[layer_id]['tile_size']
        # we have to find unique o_tiles and o_tile_size corresponding to them
        # when tiles_entire_repeat > 1, not reducing to unique set of tiles can result in overestimation of overlaps
        o_tiles_unique = []
        o_tile_size_unique = []
        for o_idx, o_tile in enumerate(o_tiles):
            if o_tile in o_tiles_unique:
                continue
            o_tiles_unique.append(o_tile)
            o_tile_size_unique.append(o_tile_size[o_idx])
        i_tile_overlap_dict = find_overlaps_multilayer(o_tiles_unique, o_tile_size_unique, i_tiles, i_tile_size)
        layer_overlap_dict[layer_id] = i_tile_overlap_dict
        
    return layer_overlap_dict, o_tiles_unique, o_tile_size_unique

def classify_dims(layer_overlap_dict, o_tiles, o_tile_size):
    dims_info = {}
    # dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    for layer_id in layer_overlap_dict.keys():
        dims_info[layer_id] = []
        # we have to identify which dimensions are 'overlap'  and which are 'complete_or_split'
        # 'overlap': among the overlaps identified, this dimension corresponds to the one where there are intersection between 
        #            non-consecutive overlaps
        for d in range(4):
            overlap_idx = []
            for overlaps in layer_overlap_dict[layer_id].keys():
                for overlap in layer_overlap_dict[layer_id][overlaps]:
                    # print(overlap)
                    o_idx = overlap[0]
                    overlap_start_idx = overlap[1]
                    overlap_end_idx = overlap[2]
                    tile_size = o_tile_size[o_idx]
                    overlap_idx.append((overlap_start_idx[d], overlap_end_idx[d]))
            unique_overlap_idx = list(set(overlap_idx))
            is_complete_or_split = True
            for i in range(len(unique_overlap_idx)-1):
                for j in range(i+1, len(unique_overlap_idx)):
                    if len(list(set(list(range(unique_overlap_idx[i][0], unique_overlap_idx[i][1]))) \
                                & set(list(range(unique_overlap_idx[j][0], unique_overlap_idx[j][1]))))) > 0:
                        is_complete_or_split = False
            if not is_complete_or_split:
                dims_info[layer_id].append(d)
    
    # permutation = []
    dims_score = {}
    for d in range(4):
        score = 0
        for layer_id in dims_info.keys():
            if d in dims_info[layer_id]:
                score += 1
        dims_score[d] = score
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    permutation = [k for k, v in sorted(dims_score.items(), key=lambda item: item[1])]
    
    return dims_info, dims_score, permutation

def search_block_multilayer(layer_tile_info_dict, layer_overlap_dict, o_tiles, o_tile_size, \
                            dims_info, dims_score, permutation, \
                            word_size=16, tag_size=64, u_multiple_of=-1):
    overhead_dict = {}
    
    # dims = get_dimension_idx(o_dimension_list, ['N', 'C', 'P', 'Q'])
    
    for layer_id in layer_overlap_dict.keys():
        results = []
        results_tag = []
        for tile_idx in layer_overlap_dict[layer_id].keys():
            for overlap in layer_overlap_dict[layer_id][tile_idx]:
                result_dict = {}
                result_dict_tag = {}
                o_idx = overlap[0]
                overlap_start_idx = overlap[1]
                overlap_end_idx = overlap[2]
                tile_size = o_tile_size[o_idx]
                
                h, w = 1, 1
                above = False
                for idx in permutation:
                    if above:
                        # overlap
                        if dims_score[idx] > 0: 
                            h *= overlap_end_idx[idx] - overlap_start_idx[idx]
                        else:
                            h *= overlap_end_idx[idx] - overlap_start_idx[idx]
                    else:
                        if dims_score[idx] > 0:
                            w *= o_tile_size[o_idx][idx]
                            above = True
                        else:
                            w *= overlap_end_idx[idx] - overlap_start_idx[idx]

                start_at_left_edge = False
                end_at_right_edge = False
                if max([dims_score[k] for k in dims_score.keys()]) == 0:
                    start_at_left_edge = True
                    end_at_right_edge = True
                    w += 1
                else:
                    dim_of_interest = -1
                    for k in permutation:
                        if dims_score[k] > 0:
                            dim_of_interest = k
                            break
                    if (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) == 0:
                        start_at_left_edge = True
                    if (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) == tile_size[dim_of_interest]:
                        end_at_right_edge = True

                overlap_size = 1
                for idx in range(4):
                    overlap_size *= overlap_end_idx[idx] - overlap_start_idx[idx]
                # print(h, w, overlap_size)
                # print(start_at_left_edge, end_at_right_edge)
                
                if u_multiple_of > 0 and w <= u_multiple_of:
                    print("Warning: the width of this tile {} is smaller than the minimum possible block size given by \
                    the constraint u_multiple_of {}. Consider rehashing this layer {} input instead of searching for optimal block \
                    assignment.".format(w, u_multiple_of, layer_id))
                    print("--", h, w, start_at_left_edge, end_at_right_edge, overlap_size, tile_idx ,o_idx)
                for u in range(1, w):
                    if u_multiple_of > 0 and w > u_multiple_of and u % u_multiple_of != 0:
                        continue
                    
                    # full tile
                    if start_at_left_edge and end_at_right_edge:
                        # for full tile, we don't have to worry about redundant read.
                        # only compute the mac tag overhead
                        reads = overlap_size
                        redundant_reads = 0
                        result_dict[u] = redundant_reads
                        result_dict_tag[u] = math.ceil(reads / u)

                    # consider the left side
                    elif start_at_left_edge and not end_at_right_edge:
                        x = int(w * (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                                tile_size[dim_of_interest])
                        reads = compute_redundant_reads(h, w, x, u, 'left')
                        redundant_reads = reads - overlap_size
                        result_dict[u] = redundant_reads
                        result_dict_tag[u] = math.ceil(reads / u)

                    # consider the right side
                    elif not start_at_left_edge and end_at_right_edge:
                        x = int(w * (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                                tile_size[dim_of_interest])
                        reads = compute_redundant_reads(h, w, x, u, 'right')
                        redundant_reads = reads - overlap_size
                        result_dict[u] = redundant_reads
                        result_dict_tag[u] = math.ceil(reads / u)
                        # if u == 162:
                        #     print(overlap_start_idx, o_tiles[o_idx], tile_size)
                        #     print(x, reads, redundant_reads)

                    # consider both the left and right
                    else:
                        x1 = int(w * (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                                 tile_size[dim_of_interest])
                        x2 = int(w * (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                                 tile_size[dim_of_interest])
                        reads = compute_redundant_reads(h, w, [x1, x2], u, 'both')
                        redundant_reads = reads - overlap_size
                        result_dict[u] = redundant_reads
                        result_dict_tag[u] = math.ceil(reads / u)

                results.append(result_dict)
                results_tag.append(result_dict_tag)
                
        u_candidates = None
        for res in results:
            if u_candidates is None:
                u_candidates = res.keys()
            else:
                u_candidates = u_candidates & res.keys()
        u_candidates = list(u_candidates)
        
        memory_read_overhead_dict = {}
        for u in u_candidates:
            overhead = 0
            # note that we don't have to compute baseline tags as we are searching overlaps for all input tiles
            for idx, res in enumerate(results):
                overhead += res[u] * word_size
            for idx, res in enumerate(results_tag):
                overhead += res[u] * tag_size
            memory_read_overhead_dict[u] = overhead
        
        overhead_dict[layer_id] = (memory_read_overhead_dict, results, results_tag)
        
    u_candidates = None
    for layer_id in overhead_dict.keys():
        if u_candidates is None:
            u_candidates = overhead_dict[layer_id][0].keys()
        else:
            u_candidates = u_candidates & overhead_dict[layer_id][0].keys()
    u_candidates = list(u_candidates)
    
    overhead_sum_dict = {}
    for layer_id in layer_overlap_dict.keys():
        for u in overhead_dict[layer_id][0]:
            if u in u_candidates:
                if u not in overhead_sum_dict:
                    overhead_sum_dict[u] = overhead_dict[layer_id][0][u]
                else:
                    overhead_sum_dict[u] += overhead_dict[layer_id][0][u]
                
    min_u = min(overhead_sum_dict, key=overhead_sum_dict.get)
    min_overhead = overhead_sum_dict[min_u]
    multilayer_result_dict = {}
    for layer_id in layer_overlap_dict.keys():
        tiles = layer_tile_info_dict[layer_id]['tiles']
        tile_size = layer_tile_info_dict[layer_id]['tile_size']
        base_read = get_baseline_reads(tiles, tile_size, [])
        tag_read = 0
        redundant_read = 0
        results, results_tag = overhead_dict[layer_id][1], overhead_dict[layer_id][2]
        for idx, res in enumerate(results):
            redundant_read += res[min_u]
        for idx, res in enumerate(results_tag):
            tag_read += res[min_u]
        multilayer_result_dict[layer_id] = {'base_read': base_read, \
                                            'redundant_read': redundant_read, \
                                            'tag_read': tag_read}
    return multilayer_result_dict, min_u

def evaluate_block_multilayer(layer_tile_info_dict, layer_overlap_dict, o_tiles, o_tile_size, \
                              dims_info, dims_score, permutation, u, \
                              word_size=16, tag_size=64):
    multilayer_result_dict = {}
    
    for layer_id in layer_overlap_dict.keys():
        redundant_read = 0
        tag_read = 0
        for tile_idx in layer_overlap_dict[layer_id].keys():
            for overlap in layer_overlap_dict[layer_id][tile_idx]:
                result_dict = {}
                result_dict_tag = {}
                o_idx = overlap[0]
                overlap_start_idx = overlap[1]
                overlap_end_idx = overlap[2]
                tile_size = o_tile_size[o_idx]
                
                h, w = 1, 1
                above = False
                for idx in permutation:
                    if above:
                        # overlap
                        if dims_score[idx] > 0: 
                            h *= overlap_end_idx[idx] - overlap_start_idx[idx]
                        else:
                            h *= overlap_end_idx[idx] - overlap_start_idx[idx]
                    else:
                        if dims_score[idx] > 0:
                            w *= o_tile_size[o_idx][idx]
                            above = True
                        else:
                            w *= overlap_end_idx[idx] - overlap_start_idx[idx]

                start_at_left_edge = False
                end_at_right_edge = False
                if max([dims_score[k] for k in dims_score.keys()]) == 0:
                    start_at_left_edge = True
                    end_at_right_edge = True
                    w += 1
                else:
                    dim_of_interest = -1
                    for k in permutation:
                        if dims_score[k] > 0:
                            dim_of_interest = k
                            break
                    if (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) == 0:
                        start_at_left_edge = True
                    # print(o_idx)
                    # print(overlap_end_idx[dim_of_interest], o_tiles[o_idx][dim_of_interest], tile_size[dim_of_interest])
                    if (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) == tile_size[dim_of_interest]:
                        end_at_right_edge = True

                overlap_size = 1
                for idx in range(4):
                    overlap_size *= overlap_end_idx[idx] - overlap_start_idx[idx]
                    
                # print(start_at_left_edge, end_at_right_edge)
                # full tile
                if start_at_left_edge and end_at_right_edge:
                    # for full tile, we don't have to worry about redundant read.
                    # only compute the mac tag overhead
                    reads = overlap_size
                    redundant_reads = 0
                    redundant_read += redundant_reads
                    tag_read += math.ceil(reads / u)
                    # print(reads, reads/u)

                # consider the left side
                elif start_at_left_edge and not end_at_right_edge:
                    x = int(w * (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                            tile_size[dim_of_interest])
                    reads = compute_redundant_reads(h, w, x, u, 'left')
                    redundant_reads = reads - overlap_size
                    redundant_read += redundant_reads
                    tag_read += math.ceil(reads / u)

                # consider the right side
                elif not start_at_left_edge and end_at_right_edge:
                    x = int(w * (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                            tile_size[dim_of_interest])
                    reads = compute_redundant_reads(h, w, x, u, 'right')
                    redundant_reads = reads - overlap_size
                    redundant_read += redundant_reads
                    tag_read += math.ceil(reads / u)
                    # print(reads, reads/u)

                # consider both the left and right
                else:
                    x1 = int(w * (overlap_start_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                             tile_size[dim_of_interest])
                    x2 = int(w * (overlap_end_idx[dim_of_interest] - o_tiles[o_idx][dim_of_interest]) / \
                             tile_size[dim_of_interest])
                    reads = compute_redundant_reads(h, w, [x1, x2], u, 'both')
                    redundant_reads = reads - overlap_size
                    redundant_read += redundant_reads
                    tag_read += math.ceil(reads / u)


        tiles = layer_tile_info_dict[layer_id]['tiles']
        tile_size = layer_tile_info_dict[layer_id]['tile_size']
        base_read = get_baseline_reads(tiles, tile_size, [])
        multilayer_result_dict[layer_id] = {'base_read': base_read, \
                                            'redundant_read': redundant_read, \
                                            'tag_read': tag_read}
        
    return multilayer_result_dict
