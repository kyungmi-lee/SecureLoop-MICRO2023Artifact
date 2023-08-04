import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class Node(object):
    def __init__(self, name, id_number, is_var):
        self.name = name
        self.id_number = id_number
        self.is_var = is_var
        self.children = []
        self.parents = []
        
    def get_children(self):
        return self.children
    
    def get_parents(self):
        return self.parents
    
class BackpropGraph(object):
    def __init__(self, model, input_size):
        self.model = model
        
        # construct the graph
        input_tensor = torch.zeros(input_size)
        self.nodes = []
        self._construct_graph(self.model, input_tensor)
        
        # store layer id - name
        self.layer_idx_to_name = {}
        self.layer_type = {}
        layer_idx = 1
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.layer_idx_to_name[layer_idx] = name
                self.layer_type[layer_idx] = 'Conv'
                layer_idx += 1
            elif isinstance(module, nn.Linear):
                self.layer_idx_to_name[layer_idx] = name
                self.layer_type[layer_idx] = 'Linear'
                layer_idx += 1
        self.n_layers = layer_idx - 1
        
    def _get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id_number == node_id:
                return node
        return ValueError("Node with id {} not found".format(node_id))
        
    def _get_root(self):
        for node in self.nodes:
            if node.name == 'output':
                return node
            
    def _get_node_by_name(self, name):
        # WARNING: if there are nodes with the same name, then the first found node will be returned
        for node in self.nodes:
            if node.name == name:
                return node
        return ValueError("Node with name {} not found".format(name))
    
    def _construct_graph(self, model, input_tensor):
        output = model(input_tensor)
        
        # get parameter names to id mapping
        params = dict(self.model.named_parameters())
        param_map = {id(v):k for k, v in params.items()}
        
        seen = set()
        
        def recurse_through_functions(fn):
            if fn in seen:
                return
            seen.add(fn)
            
            fn_name = type(fn).__name__
            self.nodes.append(Node(fn_name, str(id(fn)), False))
            
            if hasattr(fn, 'variable'):
                v = fn.variable
                seen.add(v)
                self.nodes.append(Node(param_map[id(v)], str(id(v)), True))
                self.nodes[-2].children.append(str(id(v)))
                self.nodes[-1].parents.append(str(id(fn)))
            
            if hasattr(fn, 'next_functions'):
                for f in fn.next_functions:
                    new_fn = f[0]
                    if new_fn is not None:
                        parent_node = self._get_node_by_id(str(id(fn)))
                        parent_node.children.append(str(id(new_fn)))
                        recurse_through_functions(new_fn)
                        
        root_node = Node('output', str(id(output)), False)
        seen.add(output)
        var = output.grad_fn
        root_node.children.append(str(id(var)))
        self.nodes.append(root_node)
        recurse_through_functions(var)
        
        # store parameter mapping
        self.param_map = param_map
        
        # for all nodes, check if parents are correctly recorded
        for node in self.nodes:
            for child in node.children:
                child_node = self._get_node_by_id(child)
                if node.id_number not in child_node.parents:
                    child_node.parents.append(node.id_number)
        
    def get_children(self, node, print_children=False):
        # node = self._get_node_by_id(node_id)
        children = []
        for children_id in node.children:
            children.append(self._get_node_by_id(children_id))
            
        if print_children:
            print("Node {} (id: {}) has children:".format(node.name, node.id_number))
            for children_node in children:
                print("--- Node {} (id: {})".format(children_node.name, children_node.id_number))
            if len(children) == 0:
                print("--- No children: this is leaf")
        return children
        
    def get_children_until_leaf(self, node, info_list):
        if len(node.children) == 0:
            return
        for child_id in node.children:
            child_node = self._get_node_by_id(child_id)
            info_list.append((child_node.name))
            self.get_children_until_leaf(child_node, info_list)
        
    def get_parent(self, node):
        parents = []
        for parent_id in node.parents:
            parents.append(self._get_node_by_id(parent_id))
        return parents
        
    def get_parent_until_root(self, node, info_list, curr_level_from_leaf, store_name_instead_of_dist=False):
        # print(node.name, node.id_number, node.parents)
        # print(info_list)
        if len(node.parents) == 0:
            return
        for parent_id in node.parents:
            parent_node = self._get_node_by_id(parent_id)
            if store_name_instead_of_dist:
                info_list.append((parent_id, parent_node.name))
            else:
                info_list.append((parent_id, curr_level_from_leaf))
            self.get_parent_until_root(parent_node, info_list, curr_level_from_leaf + 1, store_name_instead_of_dist)
        
    def print_from_node(self, node):
        children = self.get_children(node, True)
        for child in children:
            self.print_from_node(child)
        
    def print_graph(self):
        root_node = self._get_root()
        self.print_from_node(root_node)
        
    def find_most_recent_common_parent(self, node1_name, node2_name):
        node1 = self._get_node_by_name(node1_name)
        node2 = self._get_node_by_name(node2_name)
        
        # get list of parent information including the parent's distance from the node of our interest
        node1_parent_info = [(node1.id_number, 0)]
        self.get_parent_until_root(node1, node1_parent_info, 1)
        node2_parent_info = [(node2.id_number, 0)]
        self.get_parent_until_root(node2, node2_parent_info, 1)
        
        # find the most recent common parent
        # we can find nodes in both list (id), then sum their distance and choose the minimum one
        node1_parent_id = [x[0] for x in node1_parent_info]
        node2_parent_id = [x[0] for x in node2_parent_info]
        common_ids = list(set(node1_parent_id) & set(node2_parent_id))
        distance = []
        for common_id in common_ids:
            d = 0
            for parent in node1_parent_info:
                if parent[0] == common_id:
                    d += parent[1]
                    break
            for parent in node2_parent_info:
                if parent[0] == common_id:
                    d += parent[1]
                    break
            distance.append(d)
        min_distance_idx = distance.index(min(distance))
        recent_parent = self._get_node_by_id(common_ids[min_distance_idx])
        # print(recent_parent.name, recent_parent.id_number)
        
        return recent_parent
    
    def find_all_descendents(self, node):
        children_names = []
        self.get_children_until_leaf(node, children_names)
        return children_names
    
    def find_nodes_between_two(self, node1, node2):
        # find the path between node1 and node2, and return the name of nodes in between them
        # def isReachable(node1, node2, visited, path):
        #     visited[node1.id_number] = True
        #     path.append(node1.name)
        #     if node1 == node2:
        #         return True
        #     else:
        #         for parent in node1.parents:
        #             parent_node = self._get_node_by_id(parent)
        #             if visited[parent] == False:
        #                 if isReachable(parent_node, node2, visited, path):
        #                     return True
        #                 
        #     path.pop()
        #     return False
        def find_all_path(node1, node2, path=[]):
            path = path + [node1]
            if node1 == node2:
                return [path]
            paths = []
            for parent in node1.parents:
                parent_node = self._get_node_by_id(parent)
                if parent_node not in path:
                    newpaths = find_all_path(parent_node, node2, path)
                    for newpath in newpaths:
                        paths.append(newpath)
            return paths
        
        # visited = {}
        # for node in self.nodes:
        #     visited[node.id_number] = False
        # path = []
        # if isReachable(node1, node2, visited, path):
        #     return path[1:-1]   # remove node1 and node2
        # else:
        #     return None
        paths = find_all_path(node1, node2)
        names_in_paths = []
        for path in paths:
            name_list = []
            for node in path:
                name_list.append(node.name)
            names_in_paths.append(name_list[1:-1])
        return names_in_paths
                
    
    def isConsecutiveLayer(self, layer_idx_1, layer_idx_2):
        node1_name = self.layer_idx_to_name[layer_idx_1] + ".weight"
        node2_name = self.layer_idx_to_name[layer_idx_2] + ".weight"
        most_recent_common_parent = self.find_most_recent_common_parent(node1_name, node2_name)
        # all_descendents_name = self.find_all_descendents(most_recent_common_parent)
        # isConsecutive = True
        # for name in all_descendents_name:
        #     if 'weight' in name:
        #         if name != node1_name and name != node2_name:
        #             isConsecutive = False
        # print(most_recent_common_parent.name)
        if self.layer_type[layer_idx_2] == 'Conv' and (most_recent_common_parent.name != 'ConvolutionBackward0' \
                                                        and most_recent_common_parent.name != 'ThnnConv2DBackward'):
            return False
        if self.layer_type[layer_idx_2] == 'Linear' and most_recent_common_parent.name != 'AddmmBackward0':
            return False
        
        node1 = self._get_node_by_name(node1_name)
        node_names_in_between = self.find_nodes_between_two(node1, most_recent_common_parent)
        # node_names_in_between = [node.name for node in node_in_between]
        # print(layer_idx_1, layer_idx_2, node_names_in_between)
        # there should be only one 'MkldnnConvoultionBackward' or 'AddmmBackward' in between for consecutive layers
        exist_consecutive_path_in_least_one = False
        for node_names_in_this_path in node_names_in_between:
            if node_names_in_this_path.count('ConvolutionBackward0') + node_names_in_this_path.count('AddmmBackward0') + \
               node_names_in_this_path.count('ThnnConv2DBackward') <= 1:
                if node_names_in_this_path.count('AddBackward0') <= 1:
                    exist_consecutive_path_in_least_one = True
                # print(node_names_in_this_path)
        if not exist_consecutive_path_in_least_one:
            return False
        
        return True
        
    def isDependentLayer(self, layer_idx_1, layer_idx_2):
        node1_name = self.layer_idx_to_name[layer_idx_1] + ".weight"
        node2_name = self.layer_idx_to_name[layer_idx_2] + ".weight"
        most_recent_common_parent = self.find_most_recent_common_parent(node1_name, node2_name)
        # all_descendents_name = self.find_all_descendents(most_recent_common_parent)
        # isConsecutive = True
        # for name in all_descendents_name:
        #     if 'weight' in name:
        #         if name != node1_name and name != node2_name:
        #             isConsecutive = False
        if self.layer_type[layer_idx_2] == 'Conv' and (most_recent_common_parent.name != 'ConvolutionBackward0' \
                                                        and most_recent_common_parent.name != 'ThnnConv2DBackward'):
            return False
        if self.layer_type[layer_idx_2] == 'Linear' and most_recent_common_parent.name != 'AddmmBackward0':
            return False
        
        node1 = self._get_node_by_name(node1_name)
        node_names_in_between = self.find_nodes_between_two(node1, most_recent_common_parent)
        # node_names_in_between = [node.name for node in node_in_between]
        # there should be only one 'MkldnnConvoultionBackward' or 'AddmmBackward' in between for consecutive layers
        for node_names_in_this_path in node_names_in_between:
            if node_names_in_this_path.count('ConvolutionBackward0') + node_names_in_this_path.count('AddmmBackward0') + \
               node_names_in_this_path.count('ThnnConv2DBackward') > 1:
                return False
            if 'AddBackward0' in node_names_in_this_path: # residual connection: adding two paths
                return False

            # if two layers are consecutive, then we check if 'Pool' operation is present in between
            # otherwise, they are dependent (we consider ReLU/Batchnorm can be easily combined with convolution itself)
            for name in node_names_in_this_path:
                if 'Pool' in name:
                    return False
            
        return True
    
    def get_dependency_info(self):
        # return consecutive layer info and dependency info
        consecutive_dict = {}
        dependency_dict = {}
        for i in range(1, self.n_layers + 1):
            consecutive_dict[i] = []
            dependency_dict[i] = []
        for i in range(1, self.n_layers):
            for j in range(i + 1, self.n_layers + 1):
                isConsecutive = self.isConsecutiveLayer(i, j)
                isDependent = self.isDependentLayer(i, j)
                if isConsecutive:
                    consecutive_dict[i].append(j)
                if isDependent:
                    dependency_dict[i].append(j)
        return consecutive_dict, dependency_dict