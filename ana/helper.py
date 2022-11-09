from ana.template import Symbol, ReplacingNodeNamer
import ana.utils as utils
import json
import networkx as nx
import pandas as pd
import re


def ana_network_to_nx_graph(network):
    graph = nx.MultiDiGraph()

    def add_status(status):
        graph.add_node(status)

    def add_link(link):
        graph.add_edge(link.from_node, link.to_node)

    # Module
    for module in network.get_modules():
        graph.add_node(module)

    # Status
    for data in network.get_data():
        add_status(data)
    for goal in network.get_goals():
        add_status(goal)
    for protected_goal in network.get_protected_goals():
        add_status(protected_goal)

    # Link
    for link in network.get_conditions():
        add_link(link)
    for link in network.get_delete_links():
        add_link(link)
    for link in network.get_add_links():
        add_link(link)

    return graph


__SYMBOL_PATTERN = ReplacingNodeNamer.pattern('[^a-zA-Z]', '[A-Z]+')


def __search_symbols(name):
    symbols = re.findall(__SYMBOL_PATTERN, name)
    return sorted(set(symbols), key=symbols.index)


def __count_vars(name):
    return len(__search_symbols(name))


def __rename_vars(name):
    symbol_map = {}
    for i, symbol in enumerate(__search_symbols(name)):
        new_symbol = chr((i + 23) % 26 + 65)
        symbol_map[symbol] = new_symbol
    return utils.replace(name, __SYMBOL_PATTERN, lambda x: symbol_map[x])


def load_module_status_json(filepath, template):
    with open(filepath, mode='r', encoding='utf-8') as f:
        json_data = json.load(f)

    module_names = set()
    data_names = set()
    for module_name, links in json_data.items():
        module_names.add(module_name)
        data_names |= set(links['condition'] + links['add'] + links['delete'])

    modules = {}
    statuses = {}
    for name in module_names:
        var_count = __count_vars(name)
        module = template.add_module(var_count, __rename_vars(name))
        modules[module.name] = module

    for name in data_names:
        var_count = __count_vars(name)
        status = template.add_data(var_count, __rename_vars(name))
        statuses[status.name] = status

    def add_symbols(symbols, node_name):
        for symbol in __search_symbols(node_name):
            if symbol not in symbols:
                symbols[symbol] = Symbol(symbol)

    def get_arg_symbols(symbols, node_name):
        args = []
        for symbol in __search_symbols(node_name):
            args.append(symbols[symbol])
        return args

    def set_link(add_func, node1, node1_name, node2, node2_name):
        add_symbols(symbols, node1_name)
        add_symbols(symbols, node2_name)
        arg1 = get_arg_symbols(symbols, node1_name)
        arg2 = get_arg_symbols(symbols, node2_name)
        add_func(node1(*arg1), node2(*arg2))

    for module_name, links in json_data.items():
        symbols = {}
        module = modules[__rename_vars(module_name)]
        for data_name in links['condition']:
            status = statuses[__rename_vars(data_name)]
            set_link(template.add_condition, status, data_name, module, module_name)
        for data_name in links['add']:
            status = statuses[__rename_vars(data_name)]
            set_link(template.add_add_link, module, module_name, status, data_name)
        for data_name in links['delete']:
            status = statuses[__rename_vars(data_name)]
            set_link(template.add_delete_link, module, module_name, status, data_name)

    return modules, statuses


def load_afforance_csv(filepath, config):
    df = pd.read_csv(filepath, header=None, index_col=0)

    affordance_data = {}
    i = 0

    for column in df:
        for index in df[column][2:].index:
            if df[column][index] == '1':
                args = []
                args.append(index)
                if str(df[column]['Y']) != 'nan':
                    args.append(df[column]['Y'])
                if str(df[column]['Z']) != 'nan':
                    args.append(df[column]['Z'])
                affordance_data[i] = {
                    'module_name': __rename_vars(df[column]['動詞']),
                    'args': args
                }
                i += 1

    config.set_data(affordance_data)
    return affordance_data
