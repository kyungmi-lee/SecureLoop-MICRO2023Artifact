import numpy as np
import copy
import math
import xml.etree.ElementTree as ET
import os
import yaml
from collections import OrderedDict
from collections import Counter
from itertools import islice

# global variable
OFFSET = 0

def parse_xml_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    
    # loopnest
    topology = root.findall('engine')[0].findall('topology_')[0]
    total_levels = int(topology.findall('levels_')[0].findall('count')[0].text) - 1 # 1 level for arithmetic, we only count memory levels
    levels = topology.findall('levels_')[0].findall('item')
    loopnest_depth = 0
    loopnest = []
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
    
    # mapping
    mappings = root.findall('mapping')[0].findall('datatype_bypass_nest')[0].findall('item')
    mapping_info = []
    for idx in range(len(mappings)):
        bits = list(mappings[idx].findall('bits')[0].text)
        bits = [int(x) for x in bits]
        bits = bits[(-1 * total_levels):]
        bits = bits[::-1]
        mapping_info.append(bits)
        
    return loopnest, mapping_info

def parse_workload_file(path):
    with open(path, 'r') as f:
        workload = yaml.safe_load(f)
    dimension_list = workload['problem']['shape']['dimensions']
    stride_h = workload['problem']['instance']['Hstride']
    stride_w = workload['problem']['instance']['Wstride']
    kernel_size_r = workload['problem']['instance']['R']
    kernel_size_s = workload['problem']['instance']['S']
    padding_h = workload['problem']['instance']['Hpadding']
    padding_w = workload['problem']['instance']['Wpadding']
    N = workload['problem']['instance']['N']
    C = workload['problem']['instance']['C']
    P = workload['problem']['instance']['P']
    Q = workload['problem']['instance']['Q']
    if 'M' in workload['problem']['instance']:
        dw = False
    else:
        dw = True
    
    return dimension_list, (stride_h, stride_w), (kernel_size_r, kernel_size_s), (padding_h, padding_w), (N, C, P, Q), dw

# utils
def get_dimension_idx(dimension_list, dimension_of_interest):
    idx_list = []
    for d in dimension_of_interest:
        idx_list.append(dimension_list.index(d))
    return idx_list

def get_level(mapping):
    level = len(mapping) - 2
    while(level > -1):
        if mapping[level] == 1:
            return level
        else:
            level -= 1
    print("Error: cannot find valid level for DRAM <--> on-chip")
    return -1

def recursive_tiling_idx(info):
    seq = []
    depth = len(info)
    def recurse(depth_idx):
        curr_seq = []
        global OFFSET
        if depth_idx == 0:
            curr_seq.append(list(range(OFFSET, info[depth_idx] + OFFSET)))
            OFFSET += info[depth_idx]
            return curr_seq
        else:
            for _ in range(info[depth_idx]):
                curr_seq = recurse(depth_idx - 1)
                if curr_seq is not None:
                    seq.append(curr_seq)

    recurse(depth - 1)
    reset_offset()
    return seq

def reset_offset():
    global OFFSET
    OFFSET = 0
    
def get_tiling_info(loopnest, mapping_info, dimension_list, kernel_size_r, kernel_size_s, stride_r, stride_s, padding_r, padding_s, \
                    datatype='input', dw=False):
    # get the tiling level & prepare data
    if datatype == 'weight':
        tile_level = get_level(mapping_info[0])
        dims = get_dimension_idx(dimension_list, ['C', 'R', 'S'] if dw else ['M', 'C', 'R', 'S'])
        kernel_size = {}
        stride = {}
        padding = {}
    elif datatype == 'input':
        tile_level = get_level(mapping_info[1])
        dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
        kernel_size = {dims[2]: kernel_size_r, dims[3]: kernel_size_s}
        stride = {dims[2]: stride_r, dims[3]: stride_s}
        padding = {dims[2]: padding_r, dims[3]: padding_s}
    elif datatype == 'output':
        tile_level = get_level(mapping_info[2])
        dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'] if dw else ['N', 'M', 'P', 'Q'])
        kernel_size = {}
        stride = {}
        padding = {}
    
    #  get the loopnest above this level to determine tiling
    tile_size = {}
    num_tiles = {}
    loop_depth = {}
    repeat_indirect_valid = False
    tiles_entire_repeat = 1
    for d in dims:
        tile_size[d] = 1
        num_tiles[d] = 1
        loop_depth[d] = 0
    for idx, loop_level in enumerate(loopnest):
        for loop in loop_level:
            d = loop[0]
            start = loop[1]
            end = loop[2]
            spatial = loop[4]
            
            if idx <= tile_level:
                if d in dims:
                    tile_size[d] *= (end - start)
            else:
                if spatial > 0 and d in dims:
                    tile_size[d] *= (end - start)
                elif d in dims:
                    num_tiles[d] *= (end - start)
                    loop_depth[d] += 1
                    if (end - start) > 1 and spatial == 0:
                        repeat_indirect_valid = True
                elif spatial == 0 and repeat_indirect_valid:
                    tiles_entire_repeat *= (end - start)
                    
    # tiling start idx for each dim: since we know the tile size (and it is constant for every tiles), start idx is sufficient
    tile_repeat_info = {}
    tile_idx_info = {}
    curr_tile_level = {}
    repeat_indirect_valid = False
    for d in dims:
        tile_repeat_info[d] = [1] * (loop_depth[d] + 1)
        tile_idx_info[d] = []
        curr_tile_level[d] = 0
    for loop_level in loopnest[tile_level+1:]:
        for loop in loop_level:
            d = loop[0]
            start = loop[1]
            end = loop[2]
            spatial = loop[4]
            
            if spatial > 0:
                continue
                
            if d in dims:
                tile_idx_info[d].append(end - start)
                curr_tile_level[d] += 1
                for dd in dims:
                    if dd != d:
                        tile_repeat_info[dd][curr_tile_level[dd]] *= (end - start)
                if (end - start) > 1 and spatial == 0:
                    repeat_indirect_valid = True
            elif spatial == 0 and repeat_indirect_valid:
                for dd in dims:
                    tile_repeat_info[dd][curr_tile_level[dd]] *= (end - start)

    # compute the start idx, and repeat as necessary given by tile_repeat_info        
    tile_start_idx = {}
    for d in dims:
        if len(tile_idx_info[d]) == 0:
            tile_start_idx[d] = [0]
        elif len(tile_idx_info[d]) == 1:
            tile_start_idx[d] = list(range(0, tile_idx_info[d][0]))
        else:
            # print(tile_idx_info[d])
            tile_start_idx[d] = recursive_tiling_idx(tile_idx_info[d])
            
    # reshape tile_start_idx
    for d in dims:
        if len(tile_idx_info[d]) > 1:
            tile_start_idx[d] = np.asarray(tile_start_idx[d]).reshape(tuple(tile_idx_info[d][::-1])).tolist()
    
    for d in dims:
        new_seq = np.asarray(tile_start_idx[d])
        for axis, repeats in enumerate(tile_repeat_info[d]):
            # highest level of repeat - should be repeat-entire using np.tile
            if axis == len(tile_repeat_info[d]) - 1:
                new_shape = [1] * new_seq.ndim
                new_shape[0] = repeats
                new_seq = np.tile(new_seq, new_shape)
            else:
                new_seq = new_seq.repeat(repeats, len(tile_repeat_info[d]) - 2 - axis)
        if d in stride.keys():
            new_seq *= stride[d] * tile_size[d]
        else:
            new_seq *= tile_size[d]
        tile_start_idx[d] = new_seq.reshape(-1).tolist()
    
    if datatype == 'input':
        tile_size[dims[2]] = (tile_size[dims[2]] - 1) * stride[dims[2]] + kernel_size[dims[2]]
        tile_size[dims[3]] = (tile_size[dims[3]] - 1) * stride[dims[3]] + kernel_size[dims[3]]
    
    # for now, we will ignore loop order inside the tile 
    # TODO: does loop order inside the tile matter when we use a fine-grained buffering with buffet?
    #       (e.g. double-buffering: doesn't matter in terms of cycles)
    #       for buffets, arbitrarily changing the loop order differently from the tile's processing order
    #       might cause some difficulties in address generation at the downstream buffets? 
    """
    # finally, get the processing permutation (which dimension first)
    # innermost to outermost, dim and size
    # for spatial mapping, transpose should be applied
    processing_permutation = []
    for loop_level in loopnest[:tile_level+1]:
        for loop in loop_level:
            d = loop[0]
            start = loop[1]
            end = loop[2]
            spaital = loop[4]
            if d in dims:
                processing_permutation.append((d, end - start))
    """
    return tile_size, num_tiles, tile_start_idx, tiles_entire_repeat

# for inputs, we have to identify halos..
def is_overlap(tile1, tile2, tile_size):
    n_dim = len(tile1)
    overlap = True
    overlap_start_idx = [0] * n_dim
    overlap_end_idx = [0] * n_dim
    for i in range(n_dim):
        # end of tile1 larger than start of tile2 & start of tile1 is smaller than the end of tile2
        if tile1[i] < tile2[i] and tile2[i] < tile1[i] + tile_size[i] and tile1[i] + tile_size[i] < tile2[i] + tile_size[i]:
            overlap_start_idx[i] = tile2[i]
            overlap_end_idx[i] = tile_size[i] + tile1[i]
        elif tile2[i] < tile1[i] and tile1[i] < tile2[i] + tile_size[i] and tile2[i] + tile_size[i] < tile1[i] + tile_size[i]:
            overlap_start_idx[i] = tile1[i]
            overlap_end_idx[i] = tile_size[i] + tile2[i]
        elif tile1[i] == tile2[i]:
            overlap_start_idx[i] = tile1[i]
            overlap_end_idx[i] = tile_size[i] + tile1[i]
        else:
            overlap = False
            break
    return overlap, overlap_start_idx, overlap_end_idx
    
def is_duplicate(tile1, tile2):
    n_dim = len(tile1)
    duplicate = True
    for i in range(n_dim):
        if tile1[i] != tile2[i]:
            duplicate = False
    return duplicate
    
def find_tiles_and_halos(tile_size, num_tiles, tiles_entire_repeat, tile_idx_info, dimension_list, datatype='weight', check_halo=False, \
                         dw=False):
    #  print(datatype)
    if datatype == 'weight':
        dims = get_dimension_idx(dimension_list, ['C', 'R', 'S'] if dw else ['M', 'C', 'R', 'S'])
    elif datatype == 'input':
        dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    elif datatype == 'output':
        dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'] if dw else ['N', 'M', 'P', 'Q'])
    tiles = []
    n_tiles = 1
    for d in dims:
        n_tiles *= num_tiles[d]
    n_tiles *= tiles_entire_repeat
    for i in range(n_tiles):
        curr_tile = []
        for d in dims:
            curr_tile.append(tile_idx_info[d][i])
        tiles.append(curr_tile)
        
    halos = [] # each halo should be in format: (tile_idx_1, tile_idx_2, (start_idx_N, C, P, Q), (end_idx_N, C, P, Q))
    duplicates = {} # internal book-keeping for duplicated tiles
    for i in range(n_tiles):
        duplicates[i] = [i]

    # we only check for consecutive tiles here
    # for halos in non-consecutive tiles, we process it after removing halos from consecutive tiles
    # for consecutive tiles, we assume that halo only exists in one dimension (sliding window pattern - don't stride in two dims)
    if check_halo:
        check_until = n_tiles
        for idx1 in range(check_until - 1):
            # for idx2 in range(idx1 + 1, check_until):
            # check overlap in each dimension
            idx2 = idx1 + 1
            overlap, overlap_start_idx, overlap_end_idx = is_overlap(tiles[idx1], tiles[idx2], \
                                                                     [tile_size[dims[0]], tile_size[dims[1]], \
                                                                      tile_size[dims[2]], tile_size[dims[3]]])
            duplicate = is_duplicate(tiles[idx1], tiles[idx2])
            if duplicate:
                duplicates[idx1].append(idx2)
                duplicates[idx2].append(idx1)
            elif overlap:
                # identify in which dimension halo exists
                halo_dim = []
                for i, (s, e) in enumerate(zip(overlap_start_idx, overlap_end_idx)):
                    if e - s != tile_size[dims[i]]:
                        halo_dim.append(i)
                # check if idx1 has any duplicates: if so, only the last duplicated tile should be considered
                duplicate_idx1 = duplicates[idx1]
                if idx1 == max(duplicate_idx1):
                    halo_info = (idx1, idx2, overlap_start_idx, overlap_end_idx, halo_dim)
                    halos.append(halo_info)
    
    # for key in duplicates.keys():
    #     print(key, duplicates[key])
    return n_tiles, tiles, halos

def is_overlap_diff_size(tile1, tile2, tile_size_1, tile_size_2):
    n_dim = len(tile1)
    overlap = True
    overlap_start_idx = [0] * n_dim
    overlap_end_idx = [0] * n_dim
    for i in range(n_dim):
        # end of tile1 larger than start of tile2 & start of tile1 is smaller than the end of tile2
        if tile1[i] < tile2[i] and tile2[i] < tile1[i] + tile_size_1[i] and tile1[i] + tile_size_1[i] < tile2[i] + tile_size_2[i]:
            overlap_start_idx[i] = tile2[i]
            overlap_end_idx[i] = tile_size_1[i] + tile1[i]
        elif tile2[i] < tile1[i] and tile1[i] < tile2[i] + tile_size_2[i] and tile2[i] + tile_size_2[i] < tile1[i] + tile_size_1[i]:
            overlap_start_idx[i] = tile1[i]
            overlap_end_idx[i] = tile_size_2[i] + tile2[i]
        elif tile1[i] == tile2[i]:
            overlap_start_idx[i] = tile1[i]
            overlap_end_idx[i] = tile_size_1[i] + tile1[i]
        else:
            overlap = False
            break
    return overlap, overlap_start_idx, overlap_end_idx

def remove_halos_from_consecutive_tiles(tiles, halos, tile_size, dimension_list):
    dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    tile_size_in_list = [tile_size[d] for d in dims]
    
    halos_processed = []
    tile_size_processed = []
    for _ in range(len(tiles)):
        tile_size_processed.append(copy.deepcopy(tile_size_in_list))
        
    if len(halos) > 0:
        for halo_info in halos:
            tile1_idx = halo_info[0]
            tile2_idx = halo_info[1]
            start_idx = halo_info[2]
            end_idx = halo_info[3]
            halo_dim = halo_info[4]

            if tile2_idx - tile1_idx == 1: # consecutive tiles
                tiles[tile2_idx][halo_dim] = end_idx[halo_dim]
                tile_size_processed[tile2_idx][halo_dim] -= (end_idx[halo_dim] - start_idx[halo_dim])

    duplicates = {} # internal book-keeping for duplicated tiles
    for i in range(len(tiles)):
        duplicates[i] = [i]
    # we should only check halos for n_tiles / tiles_entire_repeat
    # after halo analysis, the result should be multiplied
    for idx1 in range(len(tiles) - 1):
        for idx2 in range(idx1 + 1, len(tiles)):
            # check overlap in each dimension
            overlap, overlap_start_idx, overlap_end_idx = is_overlap_diff_size(tiles[idx1], tiles[idx2], \
                                                                               tile_size_processed[idx1], tile_size_processed[idx2])
            duplicate = is_duplicate(tiles[idx1], tiles[idx2])
            if duplicate:
                duplicates[idx1].append(idx2)
                duplicates[idx2].append(idx1)
            elif overlap:
                # identify in which dimension halo exists
                halo_dim = []
                for i, (s, e) in enumerate(zip(overlap_start_idx, overlap_end_idx)):
                    if e - s != tile_size_processed[idx2][i]:
                        halo_dim.append(i)
                # check if idx1 has any duplicates: if so, only the last duplicated tile should be considered
                duplicate_idx1 = duplicates[idx1]
                if idx1 == max(duplicate_idx1):
                    halo_info = (idx1, idx2, overlap_start_idx, overlap_end_idx, halo_dim)
                    halos_processed.append(halo_info)
            
    return tiles, tile_size_processed, halos_processed
    
def ExtendedEuclidAlgo(a, b):
     
    # Base Case
    if a == 0 :
        return b, 0, 1
         
    gcd, x1, y1 = ExtendedEuclidAlgo(b % a, a)
     
    # Update x and y using results of recursive
    # call
    x = y1 - (b // a) * x1
    y = x1
     
    return gcd, x, y
     
# Function to give the distinct
# solutions of ax = b (mod n)
def linearCongruence(A, B, N):
    A = A % N
    B = B % N
    u = 0
    v = 0
     
    # Function Call to find
    # the value of d and u
    d, u, v = ExtendedEuclidAlgo(A, N)
     
    # No solution exists
    if (B % d != 0):
        # print(-1)
        return -1, -1
     
    # Else, initialize the value of x0
    x0 = (u * (B // d)) % N
    if (x0 < 0):
        x0 += N
     
    # Pr all the answers
    # for i in range(d):
    #     print((x0 + i * (N // d)) % N, end = " ")
    x_in_range = []
    for i in range(d):
        x_in_range.append((x0 + i * (N // d)) % N)
    x0 = min(x_in_range)
    
    return x0, d
        
def compute_redundant_reads(h, w, x, u, left_or_right='right'):
    if u >= w:
        # print(h, w)
        total_reads = (h * w)
    else:
        if h > 1:
            # we have to solve linear congruence problem satisfying the conditions
            # generate a range of possible remainders
            """
            remainders = []
            for i in range(x, w):
                if i not in remainders:
                    remainders.append(i)
                left_val = i - (u - 1)
                if left_val < 0:
                    left_val += w
                if left_val not in remainders:
                    remainders.append(left_val)

            # debug
            # print("U: {}".format(u))
            # print(remainders)
            min_r = min(remainders)
            max_r = max(remainders)
            """
            if left_or_right == 'left':
                min_r = - (u - 1)
                max_r = x - 1
            elif left_or_right == 'right':
                min_r = x - (u - 1)
                max_r = w - 1
            elif left_or_right == 'both':
                min_r = x[0] - (u - 1)
                max_r = x[1] - 1

            # for each remainder, compute the solution i for u * i = r (mod w)
            remainders_checked = []
            total_reads = 0
            for r in range(min_r, max_r+1):
                if r < 0:
                    r += w
                if r in remainders_checked:
                    continue
                else:
                    remainders_checked.append(r)
                x0, gcd = linearCongruence(u, r, w)
                if x0 > -1:
                    max_i = math.ceil(h * w / u)
                    reads = math.ceil((max_i - x0) * gcd / w) 
                    # if reads > int(reads):
                    #     reads = int(reads) + 1
                    # else:
                    #     reads = int(reads)
                    total_reads += reads * u

        # if h == 1, we don't have to run this linear congruence problem
        else:
            if left_or_right == 'left':
                total_reads = math.ceil(x / u) * u
            elif left_or_right == 'right':
                total_reads = (w - x) + (x % u)
            elif left_or_right == 'both':
                total_reads = math.ceil(x[1] / u) * u - int(x[0] / u) * u

    return total_reads

def get_baseline_reads(tile, tile_size, halo):
    # dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    reads = 0
    for tile_idx in range(len(tile)):
        base_read = 1
        for ts in tile_size[tile_idx]:
            base_read *= ts
        for h in halo:
            if h[1] == tile_idx and h[0] == tile_idx - 1: # only for the consecutive tiles
                halo_size = 1
                for (s, e) in zip(h[2], h[3]):
                    halo_size *= (e - s)
                base_read -= halo_size
        reads += base_read
    return reads
    
# def get_tag_counts(tile, tile_size, halo, u, dimension_list):
#     dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
#     num_tags = 0
#     for tile_idx in range(len(tile)):
#         base_read = 1
#         for ts in tile_size[tile_idx]:
#             base_read *= ts
#         for h in halo:
#             if h[1] == tile_idx: # this should be for all tiles
#                 halo_size = 1
#                 for (s, e) in zip(h[2], h[3]):
#                     halo_size *= (e - s)
#                 base_read -= halo_size
#         num_tags += math.ceil(base_read / u)
        
#     return num_tags

def get_tag_counts(tile_for_tag_count, tile, tile_size, u):
    num_tags = 0
    for tile_idx in range(len(tile)):
        base_read = 1
        for (t, ts, tt) in zip(tile[tile_idx], tile_size[tile_idx], tile_for_tag_count[tile_idx]):
            base_read *= (ts - (tt - t))
        num_tags += math.ceil(base_read / u)
    return num_tags
            
def process_halos(tiles, halo):
    processed_halo = []
    processed_tiles_for_tag_compute = copy.deepcopy(tiles)
    for halo_info in halo:
        tile1_idx = halo_info[0]
        tile2_idx = halo_info[1]
        start_idx = halo_info[2]
        end_idx = halo_info[3]
        halo_dim = halo_info[4]
        if tile2_idx - tile1_idx == 1:
            continue
        start_idx_relative = (np.asarray(start_idx)-np.asarray(processed_tiles_for_tag_compute[tile1_idx])).tolist()
        end_idx_relative = (np.asarray(end_idx)-np.asarray(processed_tiles_for_tag_compute[tile1_idx])).tolist()
        complete_overlap = True
        for hd in halo_dim:
            if start_idx_relative[hd] >= 0:
                complete_overlap = False
        if not complete_overlap and  min(start_idx_relative) < 0:
            start_idx_relative = np.asarray(start_idx_relative)
            start_idx_relative[start_idx_relative < 0] = 0
            start_idx_relative = start_idx_relative.tolist()
        halo_size = 1
        for (s, e) in zip(start_idx_relative, end_idx_relative):
            halo_size *= (e - s)
        
        if not complete_overlap and halo_size > 0:
            new_halo_info = (start_idx_relative, end_idx_relative, halo_dim)
            processed_halo.append(new_halo_info)
            for hd in halo_dim:
                # tile_size_processed[tile2_idx][hd] -= (end_idx[hd] - tiles[tile2_idx][hd])
                processed_tiles_for_tag_compute[tile2_idx][hd] = end_idx[hd]
        elif complete_overlap:
            for hd in halo_dim:
                # tile_size_processed[tile2_idx][hd] -= (end_idx[hd] - tiles[tile2_idx][hd])
                processed_tiles_for_tag_compute[tile2_idx][hd] = end_idx[hd]
            
    unique_halo = []
    unique_halo_counts = []
    for h in processed_halo:
        if h in unique_halo:
            unique_idx = unique_halo.index(h)
            unique_halo_counts[unique_idx] += 1
        else:
            unique_halo.append(h)
            unique_halo_counts.append(1)
            
    return processed_tiles_for_tag_compute, unique_halo, unique_halo_counts

"""
# For each datatype we have tiles and halos information from above
# then, with the user-defined variable for the AES-GCM block size (e.g., AES: 128bit, GCM: 512bit), we can count: 
# - additional memory traffic (redundant read + MAC tags)
# - AES encryption/decryption counts
# - GF multiplication counts
# - XOR counts
def count_actions(tiles, halos, n_tiles, tiles_entire_repeat, tile_size, word_size, tag_size, aes_block_size, mac_block_size, dimension_list, \
                  datatype='weight', no_halo=True):
    # if there's no halo, then each tile has no overlap and has same size
    # analysis on a single tile is sufficient to compute the counts for the entire datatype
    # in a single tile we first count how many MAC blocks are present within a tile
    # from there, we can compute AES blocks and MAC tag counts
    # also, can count AES encryption/decryption counts, GF mult counts, and XOR counts
    # - for encryption: counter should be encrypted with AES to geterate OTP, XOR with plaintext to obtain ciphertext
    # - for decryption: vice versa, the OTP can be XOR with the ciphertext
    # - for authentication: GF mult for encrypted data, XOR with the next block, GF mult... 
    # - TODO: for lightweight crypto functions / other hashes, these counts should be adjusted
    if datatype == 'weight':
        dims = get_dimension_idx(dimension_list, ['M', 'C', 'R', 'S'])
    elif datatype == 'input':
        dims = get_dimension_idx(dimension_list, ['N', 'C', 'P', 'Q'])
    elif datatype == 'output':
        dims = get_dimension_idx(dimension_list, ['N', 'M', 'P', 'Q'])
    
    # values to return
    baseline_word_read = 0   # count (words)
    baseline_word_write = 0  # count (words)
    redundant_word_read = 0  # count (words)
    mac_tags_read = 0        # count (words)
    mac_tags_write = 0       # count (words)
    
    total_read_bits = 0      # bits
    total_write_bits = 0     # bits
    
    aes_engine_count = 0     # count (128bit datapath)
    gf_mult_count = 0        # count (128bit datapath)
    xor_count = 0            # count (128bit datapath)
    
    if no_halo:
        tile_size_scalar = 1
        for d in dims:
            tile_size_scalar *= tile_size[d]
        n_blocks_in_tile = math.ceil(tile_size_scalar / (mac_block_size / word_size))
        words_in_block = mac_block_size / word_size
        if datatype == 'weight' or datatype == 'input': # read-only datatypes
            baseline_word_read = tile_size_scalar * n_tiles
            mac_tags_read = n_blocks_in_tile * n_tiles
        elif datatype == 'output': # read-write datatype
            baseline_word_write = tile_size_scalar * n_tiles
            mac_tags_write = n_blocks_in_tile * n_tiles
            # if tile_entire_repeat > 0: 
            baseline_word_read = tile_size_scalar * (n_tiles / tiles_entire_repeat) * (tiles_entire_repeat - 1)
            mac_tags_read = n_blocks_in_tile * (n_tiles / tiles_entire_repeat) * (tiles_entire_repeat - 1)
    
    # when there are halos, we have to compute additional overhead from redundant reads
    # we only consider halos in input feature map
    else:
        # if we have halos, we should first remove halos from consecutive tiles
        tiles_processed, tile_size_processed, halos_processed = remove_halos_from_consecutive_tiles(copy.deepcopy(tiles), halos, \
                                                                                                    copy.deepcopy(tile_size), dimension_list)
        processed_halo, unique_halo, unique_halo_counts = process_halos(copy.deepcopy(tiles_processed), halos_processed)
        # if len(unique_halo) == 0, we don't have to worry about redundant read
        if len(unique_halo) > 0:
            # determine the permutation order for the tiles
            # intuitively, the permutation order should place the halo dim at the outermost loop, 
            # so that overlapping blocks can be minimized
            halo_dims = [h_dim for halo in unique_halo for h_dim in halo[2]]
            # max_count_halo_idx = unique_halo_counts.index(max(unique_halo_counts))
            halo_dims_count = Counter(halo_dims)
            max_count_halo_dim = max(halo_dims_count, key=halo_dims_count.get)
            permutation = []
            for idx, d in enumerate(dims):
                if idx not in halo_dims:
                    permutation.append(d)
            for idx, d in enumerate(dims):
                if idx != max_count_halo_dim and idx in halo_dims:
                    permutation.append(d)
            permutation.append(dims[max_count_halo_dim])

            results_redundant_read = []
            results_additional_tag_read = []
            for halo in unique_halo:
                # process halo in 2-d shape using processing_permutation
                halo_start_idx = halo[0]
                halo_end_idx = halo[1]
                halo_dim_list = halo[2]
                if len(halo_dim_list) == 1:
                    halo_dim = halo_dim_list[0]
                    h, w = 1, 1
                    above = False
                    for p in permutation:
                        if p == dims[halo_dim]:
                            w *= halo_end_idx[dims.index(p)]
                            above = True
                        elif not above:
                            w *= halo_end_idx[dims.index(p)]
                        else:
                            h *= halo_end_idx[dims.index(p)]
                            
                    x = int(w * halo_start_idx[halo_dim] / halo_end_idx[halo_dim])
                    halo_size = 1
                    for (s, e) in zip(halo_start_idx, halo_end_idx):
                        halo_size *= (e - s)
                else:
                    # when there are halos over multiple dimensions (assuming max 2 dimensions for 2d conv) 
                    # we will ignore a dimension corresponding to max_count_halo_dim 
                    halo_dim_translated = [dims[x] for x in halo_dim_list if x != max_count_halo_dim]
                    h, w = 1, 1
                    above = False
                    for p in permutation:
                        if p in halo_dim_translated:
                            w *= halo_end_idx[dims.index(p)]
                            above = True
                        elif p == dims[max_count_halo_dim]:
                            h *= (halo_end_idx[dims.index(p)] - halo_start_idx[dims.index(p)])
                        elif not above:
                            w *= halo_end_idx[dims.index(p)]
                        else:
                            h *= halo_end_idx[dims.index(p)]
                    x = int(w * halo_start_idx[dims.index(halo_dim_translated[0])] / halo_end_idx[dims.index(halo_dim_translated[0])])
                    halo_size = 1
                    for (s, e) in zip(halo_start_idx, halo_end_idx):
                        halo_size *= (e - s)

                u = mac_block_size // word_size
                reads = compute_redundant_reads(h, w, x, u)
                redundant_reads = reads - halo_size
                results_redundant_read.append(redundant_reads)
                results_additional_tag_read.append(math.ceil(reads / u))

            baseline_word_read = get_baseline_reads(tiles_processed, tile_size_processed, halos_processed, dimension_list)
            mac_tags_read = get_tag_counts(tiles_processed, tile_size_processed, halos_processed, mac_block_size // word_size, dimension_list)
            for idx in range(len(unique_halo)):
                mac_tags_read += results_additional_tag_read[idx] * unique_halo_counts[idx]
                redundant_word_read += results_redundant_read[idx] * unique_halo_counts[idx]
                
        else:
            baseline_word_read = get_baseline_reads(tiles_processed, tile_size_processed, halos_processed, dimension_list)
            mac_tags_read = get_tag_counts(tiles_processed, tile_size_processed, halos_processed, mac_block_size // word_size, dimension_list)
            
    # counts for AES, GFMult, and XOR
    # Encryption (word write): generate OTP with AES (128bit), and XOR with plaintext
    # Decryption (word read): generate OTP with AES (128bit), and XOR with ciphertext
    # Authentication: (GF multiplication (128bit datapath) and XOR) * mac_block_size / aes_block_size + 1 AES
    
    # TODO: For redundant read, should we also count decryption for redundant reads (technically they don't need to be decrypted?)
    
    # assume every block will be encrypted/decrypted + 1 AES for authentication
    aes_engine_count = mac_tags_read * (mac_block_size / aes_block_size + 1)  + \
                       mac_tags_write * (mac_block_size / aes_block_size  + 1) 
    gf_mult_count = mac_tags_read * (mac_block_size / aes_block_size)  + \
                    mac_tags_write * (mac_block_size / aes_block_size) 
    xor_count = 2 * (mac_tags_read * (mac_block_size / aes_block_size) + mac_tags_write * (mac_block_size / aes_block_size))
    
    total_read_bits = (baseline_word_read + redundant_word_read) * word_size + mac_tags_read * tag_size
    total_write_bits = baseline_word_write * word_size + mac_tags_write * tag_size
    
    return baseline_word_read, baseline_word_write, redundant_word_read, mac_tags_read, mac_tags_write, \
            aes_engine_count, gf_mult_count, xor_count, total_read_bits, total_write_bits

"""
    