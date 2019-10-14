import itertools as it
import os
from collections import defaultdict, Counter, deque
from copy import deepcopy
from time import time

import networkx as nx
from sortedcontainers import SortedList


def convert_VE(id_edge_map):
    # new Graph <- reverse vertices with edges
    vedge_map = defaultdict(SortedList)
    for edge_id, (l, r) in id_edge_map.items():
        vedge_map[l].add(edge_id)
        vedge_map[r].add(edge_id)

    vedges = list(it.chain(*(it.combinations(vedges, 2) for vedges in vedge_map.values())))
    return vedges


def convert_EPgraph(edge_path_ids):
    # new Paths <- based on path intersection
    paths_no_intersec = set()
    new_edges = list()
    for v, path_ids in edge_path_ids.items():
        if len(path_ids) > 1:
            new_edges += list(it.combinations(sorted(path_ids), 2))
        else:
            paths_no_intersec |= path_ids
    paths_no_intersec -= set(x for t in new_edges for x in t)
    return new_edges, paths_no_intersec


mean = lambda l: sum(l) / len(l)


def mean_cut(l, cut=1):
    if len(l) > cut * 2 + 1:
        return mean(sorted(l)[cut:-cut])
    return mean(l)


def read_data(fld, use_topologyE=True):
    fld = os.path.join('inputs', fld)
    path_params_dict = {}

    # read newrouting
    edge_path_ids = defaultdict(set)
    path_bw_max = dict()
    p_id_list = []
    p_edges = set()
    with open(os.path.join(fld, 'newrouting.csv'), 'r') as f:
        for x in map(lambda _: _.split(','), f.readlines()[1:]):
            path_id, edge_id, bw = int(x[0]), int(x[1]), int(x[2])
            edge_path_ids[edge_id].add(path_id)
            path_bw_max[path_id] = bw
            p_id_list.append(path_id)
            p_edges.add(edge_id)

    path_length_map = Counter(p_id_list)

    if use_topologyE:

        # read links
        id_edge_map = dict()
        with open(os.path.join(fld, 'links.csv'), 'r') as f:
            for row in map(lambda _: _.split(','), f.readlines()[1:]):
                a, b = int(row[1]), int(row[2])
                edge = (min(a, b), max(a, b))
                id_edge_map[int(row[0])] = edge
        # link_max_id = max(id_edge_map.keys())
        # print('link_max_id', link_max_id)

        # remove unused edges
        rem_edges = id_edge_map.keys() - p_edges
        for edge_id in rem_edges:
            id_edge_map.pop(edge_id, None)

        vedges = convert_VE(id_edge_map)
        gE = nx.Graph()
        gE.add_edges_from(vedges)
        # centralityE = nx.closeness_centrality(gE, distance=3)
        # phi = (1 + math.sqrt(2)) / 5.
        # centralityE = nx.katz_centrality_numpy(gE, 1/phi)
        # centralityE = nx.pagerank(gE, alpha=0.5)
        centralityE = nx.betweenness_centrality(gE, seed=113)

        path_centrality_sum_map = defaultdict(list)
        for e, cent in centralityE.items():
            for p_id in edge_path_ids[e]:
                path_centrality_sum_map[p_id].append(cent)

        for p_id, cent_sum in deepcopy(path_centrality_sum_map).items():
            # ToDO
            path_centrality_sum_map[p_id] = mean_cut(cent_sum, 1)

        path_params_dict['avg_centrality'] = path_centrality_sum_map

    # print(list(path_centrality_sum_map.items()))

    new_edges, paths_no_intersec = convert_EPgraph(edge_path_ids)

    # read Path G
    gP = nx.Graph()
    gP.add_edges_from(new_edges)

    # centralityP = nx.closeness_centrality(gP, distance=3)
    # path_params_dict['centrality'] = centralityP
    path_params_dict.update({'length': path_length_map, 'bw': path_bw_max})
    return gP, paths_no_intersec, path_params_dict


def strategy_degree_less_first(g, pseudo=None, pseudo2=None):
    return sorted(g, key=g.degree, reverse=False)


def strategy_other(g, path_prop_map, pseudo=None):
    return sorted(g, key=lambda _: path_prop_map.get(_, 0), reverse=False)


def strategy_combined_bw_other(g, path_param_map_list, t):
    a = path_param_map_list[0]  # bw
    b = path_param_map_list[1]  # other

    if len(path_param_map_list) > 2:
        c = path_param_map_list[2]
        key_f = lambda _: ((a[_] if a[_] < t else t) * b.get(_, 0) * c[_])
    else:
        key_f = lambda _: ((a[_] if a[_] < t else t) * b.get(_, 0))
    return sorted(g, key=key_f, reverse=False)


def strategy_combined_bw_degree(g, path_bw_map):
    return sorted(g, key=lambda _: (path_bw_map[_], g.degree(_)), reverse=False)


def intersect(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def not_overlaps(interval, other_intervals):
    return all(not intersect(interval, other) for other in other_intervals)


def is_empty(generator):
    for _ in generator:
        return False
    return True


class Coloring:
    """
    v - is edge_id
    """

    def __init__(self, path_params_dict):
        self.COLOR_MAX = 320  # 0...319
        self.path_bw_max = path_params_dict['bw']
        self.strategies = [lambda _, th=22: f(_, p, th) for f, p in [
            # (strategy_degree_less_first, None),

            # (strategy_other, path_params_dict['bw']),
            # (strategy_other, path_params_dict['length']),
            # (strategy_other, path_params_dict['avg_centrality']),

            # (strategy_combined_bw_other, (self.path_bw_max, path_params_dict['length'])),
            (strategy_combined_bw_other, (self.path_bw_max, path_params_dict['avg_centrality'])),
            # (strategy_combined_bw_other, (self.path_bw_max, path_params_dict['avg_centrality'], path_params_dict['length'])),
            # (strategy_combined_bw_other, (self.path_bw_max, path_params_dict['length'], path_params_dict['avg_centrality'])),
            # (strategy_combined_bw_other, self.path_bw_max),
        ]]

    def coloring(self, g, strategy, th):
        nodes = strategy(g, th)
        v_interval_map = dict()
        for u in nodes:
            # Set to keep track of colors of neighbours
            neighbour_intervals = {v_interval_map[v] for v in g[u] if v in v_interval_map}
            # Find the first unused interval.
            bw = self.path_bw_max[u]
            tryes = self.COLOR_MAX - bw
            i = 0
            while i <= tryes:
                interval = (i, i + bw - 1)
                if not_overlaps(interval, neighbour_intervals):
                    break
                else:
                    i = max(_[1] for _ in filter(lambda x: intersect(interval, x), neighbour_intervals)) + 1

            if i > tryes:
                continue
            # Assign the new color to the current node.
            v_interval_map[u] = interval
        return v_interval_map

    def coloring2(self, g, strategy, th):
        nodes = deque(strategy(g, th))
        v_interval_map = dict()
        while len(nodes) > 0:
            u = nodes.popleft()
            # Set to keep track of colors of neighbours
            neighbour_intervals_map = {v_interval_map[v]: v for v in g[u] if v in v_interval_map}
            neighbour_intervals = set(neighbour_intervals_map.keys())
            # Find the first unused interval.
            bw = self.path_bw_max[u]
            tryes = self.COLOR_MAX - bw
            i = 0
            while i <= tryes:
                interval = (i, i + bw - 1)
                if not_overlaps(interval, neighbour_intervals):
                    break
                else:
                    i = max(_[1] for _ in filter(lambda x: intersect(interval, x), neighbour_intervals)) + 1

            if i > tryes:
                intersected_neighbors = lambda: filter(lambda x: intersect(interval, x), neighbour_intervals)
                to_remove = []

                if bw >= 10:
                    continue
                while max(((_[1] - _[0] + 1) for _ in intersected_neighbors()), default=0) >= 16 and is_empty(filter(lambda x: (x[1] - x[0] + 1) < bw, intersected_neighbors())):
                    interval_remove = max(intersected_neighbors(), key=lambda _: (_[1] - _[0] + 1))
                    neighbour_intervals.remove(interval_remove)
                    path_id = neighbour_intervals_map[interval_remove]
                    v_interval_map.pop(path_id)
                    to_remove.append(path_id)
                    break
                if to_remove:
                    to_remove.append(u)
                    sorted_remove = sorted(to_remove, key=self.path_bw_max.get, reverse=True)
                    l_th = 20
                    for p_id in filter(lambda _: self.path_bw_max[_] < l_th, sorted_remove):
                        nodes.appendleft(p_id)

                    # sorted_remove_big = list(filter(lambda _: self.path_bw_max[_] >= l_th, sorted_remove))
                    # sorted_remove_big.reverse()
                    # for p_id in sorted_remove_big:
                    #     nodes.append(p_id)

                # print(to_remove, [self.path_bw_max[x] for x in to_remove])
                # nodes.appendleft(u)
                continue
            # Assign the new color to the current node.
            v_interval_map[u] = interval
        return v_interval_map

    def greed_search(self, g):
        max_degree = lambda: max({x[1] for x in g.degree})
        best_count = 0
        best_v_interval_map = None
        deg_th = 128

        # remove before
        for _ in range(1):
            deg_filter_th = max_degree()
            if deg_filter_th < deg_th + 1:
                break
            to_remove = {v for v, deg in g.degree if deg >= deg_filter_th}
            g.remove_nodes_from(to_remove)
        i = 0
        deg_filter_th = max_degree()
        if deg_filter_th < deg_th:
            deg_th = deg_filter_th

        if deg_filter_th < 130:
            deg_filter_th += 1

        start_t = time()

        while deg_filter_th >= deg_th:
            local_best_st = -1
            local_best_th = -1
            local_best_count = 0

            to_remove = {v for v, deg in g.degree if deg >= deg_filter_th}
            g.remove_nodes_from(to_remove)

            for st_i, strategy in enumerate(self.strategies):
                for th in range(14, max(self.path_bw_max.values()) - 2 * 3 + 1, 2):
                    # ToDo
                    v_interval_map = self.coloring(g, strategy, th)
                    count_colored = len(v_interval_map)

                    if count_colored >= local_best_count:
                        local_best_count = count_colored
                        local_best_st = st_i
                        local_best_th = th

                    if count_colored > best_count:
                        best_v_interval_map = v_interval_map
                        best_count = count_colored
            print('{}\tdeg_filter: {}, strategy: {}, |V|: {}, th: {}'.format(local_best_count, deg_filter_th, local_best_st, nx.number_of_nodes(g), local_best_th))
            deg_filter_th = max_degree()
            i += 1
            if i==5:
                break



        elapsed_time = time() - start_t
        print("Time elapsed: {:.3f} seconds".format(elapsed_time))

        return best_v_interval_map


def show_count_colored(fld, fname='coloring.csv'):
    fld = os.path.join('inputs', fld, fname)
    with open(fld, 'r') as f:
        print('============\nBest_ans:', sum(1 for _ in f.readlines()[1:] if int(_.split(',')[1]) >= 0))


def dump_results(fld, v_interval_map, path_bw_max):
    f_path = os.path.join('inputs', fld, 'coloring.csv')
    with open(f_path, 'w') as f:
        f.write('path_id,min_slice\n')
        vs = sorted(path_bw_max.keys())
        print('MIN v_interval_map: {}, count: {}'.format(min(_[0] for _ in v_interval_map.values()), len(v_interval_map)))
        f.write('\n'.join("{},{}".format(v, v_interval_map.get(v, (-1,))[0]) for v in vs))
    print('dumped to', f_path)


def execute_one(fld='49'):
    g, paths_no_intersec, path_params_dict = read_data(fld)
    g_check = nx.Graph.copy(g)
    print('nodes_count', nx.number_of_nodes(g))
    print('paths_no_intersec', paths_no_intersec)

    # centrality = nx.eigenvector_centrality(g)
    # print(sorted(centrality.items(), key=lambda _: _[1], reverse=True))
    # th = 0.0198973135
    # th = 0.099
    # to_remove = [n for n, cent in centrality.items() if cent > th]
    # 130
    # to_remove = {v for v, deg in nx.degree(g) if deg >= 130}
    # g.remove_nodes_from(to_remove)
    # print('nodes_count', nx.number_of_nodes(g))

    # smallest_last
    # d = nx.coloring.greedy_color(g, strategy='smallest_last')
    # max_d = max(d.values())
    # print('max_d', max_d)
    # to_remove = {v for v, col in d.items() if col >= max_d-2}
    # g.remove_nodes_from(to_remove)
    # print('nodes_count', nx.number_of_nodes(g))

    # degree analysis
    # print(sorted(g.degree, key=lambda x: x[1], reverse=True))

    # # Color all components
    coloring = Coloring(path_params_dict=path_params_dict)
    best_v_interval_map = coloring.greed_search(g)

    # show_count_colored('49') # 183
    best_v_interval_map.update({_: (0, path_params_dict['bw'][_] - 1) for _ in paths_no_intersec if _ not in best_v_interval_map})
    print('Final', len(best_v_interval_map))
    check_results(g_check, best_v_interval_map, path_params_dict['bw'])
    # dump_results(fld, best_v_interval_map, path_params_dict['bw'])


def execute_all(remove_prev=True):
    fld_list = sorted(filter(str.isdigit, os.listdir('./inputs')))
    fld_list = filter(lambda _: int(_) >= 48, fld_list)

    for fld in fld_list:
        coloring_f_prev = os.path.join('inputs', fld, 'coloring_prev.csv')
        if remove_prev and os.path.exists(coloring_f_prev):
            os.remove(coloring_f_prev)

        coloring_f = os.path.join('inputs', fld, 'coloring.csv')
        if os.path.exists(coloring_f):
            show_count_colored(fld)
            if not remove_prev:
                os.rename(coloring_f, coloring_f_prev)
            print('{} moved'.format(fld))
        print("\n---- {} ----".format(fld))
        execute_one(fld)


def show_all():
    fld_list = sorted(filter(str.isdigit, os.listdir('./inputs')))
    for fld in fld_list:
        show_count_colored(fld)
        print('fld', fld)


def check_results(g, best_v_interval_map, path_bw_max):
    for u in sorted(path_bw_max.keys()):
        interval_u = best_v_interval_map.get(u, -1)
        if interval_u == -1:
            continue
        if u not in g:
            print(u, 'not in g')
            continue
        for v in g[u]:
            interval_v = best_v_interval_map.get(v, -1)
            if interval_v == -1:
                continue
            if intersect(interval_u, interval_v):
                print("Error!: {}-{} on interval_a: {}, interval_b: {}".format(u, v, interval_u, interval_v))


if __name__ == '__main__':
    execute_one('53')
    # execute_all()
    # show_count_colored('49')
    # show_all()
