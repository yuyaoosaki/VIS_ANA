from abc import ABCMeta, abstractmethod
from ana.model import ANANetwork
from enum import Enum, auto
from itertools import permutations
import ana.utils as utils
import re
import logging

logger = logging.getLogger('ana')


class TemplateError(Exception):
    pass


class NodeType(Enum):
    Module = auto()
    Data = auto()
    Goal = auto()
    ProtectedGoal = auto()


class LinkType(Enum):
    Condition = auto()
    Delete = auto()
    Add = auto()


class ExpansionConfigurator(metaclass=ABCMeta):
    def __init__(self, node_namer=None):
        if node_namer is None:
            self.__node_namer = StandardNodeNamer()
        elif isinstance(node_namer, NodeNamer):
            self.__node_namer = node_namer
        else:
            raise TemplateError('Invalid node_namer. It must be NodeNamer')

    @property
    def node_namer(self):
        return self.__node_namer

    @node_namer.setter
    def node_namer(self, node_namer):
        if not isinstance(node_namer, NodeNamer):
            raise TemplateError('Invalid node_namer. It must be NodeNamer')
        self.__node_namer = node_namer

    def name(self, var_node, things):
        return self.__node_namer.name(var_node.name, *things)

    @abstractmethod
    def can_be_expanded(self, var_node, things):
        pass

    @abstractmethod
    def threshold(self, var_node, things):
        pass

    @abstractmethod
    def function(self, var_node, things):
        pass

    @abstractmethod
    def intensity(self, var_node, things):
        pass

    def goal_intensity(self, var_node, things):
        """
        Goal からの伝播で使用する intensity を個別に定義したい場合はこのメソッドをオーバーライドする
        """
        return self.intensity(var_node, things)

    def protected_goal_intensity(self, var_node, things):
        """
        ProtectedGoal からの伝播で使用する intensity を個別に定義したい場合はこのメソッドをオーバーライドする
        """
        return self.intensity(var_node, things)


class NodeNamer(metaclass=ABCMeta):
    """
    VarNode を Node に展開するときの識別名を決定する
    """

    @abstractmethod
    def name(self, var_name, things):
        pass


class StandardNodeNamer(NodeNamer):
    """
    引数を括弧で囲う

    Example
    -------
    Observe_X に X = cup を代入するとき Observe_X(cup) になる
    """

    def name(self, var_name, *things):
        if not things:
            return var_name
        return var_name + '(' + ', '.join(map(str, things)) + ')'


class ReplacingNodeNamer(NodeNamer):
    """
    変数を引数で置換する

    Example
    -------
    Observe_X に X = cup を代入するとき Observe_cup になる
    """

    def __init__(self, sep='[^a-zA-Z]', symbol='[A-Z]+'):
        super().__init__()
        self.__pattern = ReplacingNodeNamer.pattern(sep, symbol)

    @staticmethod
    def pattern(sep, symbol):
        pattern = '(?:^|(?<=' + sep + '))'
        pattern += '(' + symbol + ')'
        pattern += '(?:$|(?=' + sep + '))'
        return pattern

    def name(self, var_name, *things):
        symbols = re.findall(self.__pattern, var_name)
        symbol_map = {}
        for old, new in zip(symbols, things):
            symbol_map[old] = new
        return utils.replace(var_name, self.__pattern, lambda x: symbol_map[x])


class TemplateNetwork:
    def __init__(self, network, configurator):
        if not isinstance(network, ANANetwork):
            raise TemplateError('Invalid network. It must be ANANetwork')
        if not isinstance(configurator, ExpansionConfigurator):
            raise TemplateError('Invalid configurator. It must be ExpansionConfigurator')
        self.__network = network
        self.__config = configurator
        self.__groups = []
        self.__linked = {link_type: [] for link_type in LinkType}

    def add_module(self, var_count=0, name=None):
        node = VarNode.module(var_count, name)
        # logger.info(f'New Variable Module: {node.name}')
        return node

    def add_status(self, var_count=0, name=None):
        return self.add_data(var_count, name)

    def add_data(self, var_count=0, name=None):
        node = VarNode.data(var_count, name)
        # logger.info(f'New Variable Status: {node.name}')
        return node

    def add_goal(self, var_count=0, name=None):
        node = VarNode.goal(var_count, name)
        # logger.info(f'New Variable Goal: {node.name}')
        return node

    def add_protected_goal(self, var_count=0, name=None):
        node = VarNode.protected_goal(var_count, name)
        # logger.info(f'New Variable Protected Goal: {node.name}')
        return node

    def add_condition(self, from_node, to_node):
        self.__add_link(from_node, to_node, LinkType.Condition)

    def add_delete_link(self, from_node, to_node):
        self.__add_link(from_node, to_node, LinkType.Delete)

    def add_add_link(self, from_node, to_node):
        self.__add_link(from_node, to_node, LinkType.Add)

    def __add_link(self, from_node, to_node, link_type):
        if not isinstance(from_node, BindedVarNode):
            raise TemplateError('Invalid from_node. It must be BindedVarNode')
        if not isinstance(to_node, BindedVarNode):
            raise TemplateError('Invalid to_node. It must be BindedVarNode')
        group = {
            'links': [
                {
                    'from': from_node,
                    'to': to_node,
                    'type': link_type,
                }
            ],
            'symbols': set(),
        }
        for node in [from_node, to_node]:
            for symbol in node.symbols:
                group['symbols'].add(symbol)
        self.__groups.append(group)

        # 同一シンボルを含むグループを１つにまとめる
        new_groups = self.__groups[:]
        old_groups = []
        while len(new_groups) != len(old_groups):
            old_groups = new_groups
            new_groups = []
            for group1 in old_groups:
                for group2 in new_groups:
                    if len(group1['symbols'] & group2['symbols']) == 0:
                        continue
                    group2['symbols'] |= group1['symbols']
                    group2['links'] += group1['links']
                    break
                else:
                    new_groups.append(group1)
        self.__groups = new_groups

    def expand(self, *things):
        self.__network.stop_notifying()

        def check(link, symbols, direction):
            things = []
            for symbol in link[direction].symbols:
                i = symbols.index(symbol)
                things.append(part_things[i])
            if self.__config.can_be_expanded(link[direction].var_node, things):
                return things
            return None

        for group in self.__groups:
            symbols = list(group['symbols'])
            for part_things in permutations(things, len(symbols)):
                for link in group['links']:
                    from_things = check(link, symbols, 'from')
                    if from_things is None:
                        continue
                    to_things = check(link, symbols, 'to')
                    if to_things is None:
                        continue
                    to_node = link['to'].var_node.expand(self.__network, self.__config, *to_things)
                    from_node = link['from'].var_node.expand(self.__network, self.__config, *from_things)
                    self.__link_nodes(from_node, to_node, link['type'])
        self.__network.start_notifying()

    def __link_nodes(self, from_node, to_node, link_type):
        if (from_node, to_node) in self.__linked[link_type]:
            return
        if link_type is LinkType.Condition:
            self.__network.create_condition(from_node, to_node)
        elif link_type is LinkType.Delete:
            self.__network.create_delete_link(from_node, to_node)
        elif link_type is LinkType.Add:
            self.__network.create_add_link(from_node, to_node)
        self.__linked[link_type].append((from_node, to_node))


class VarNode:
    __number_of_nodes = 0

    def __init__(self, node_type, var_count, name=None):
        if not isinstance(node_type, NodeType):
            raise TemplateError('Invalid node_type. It must be NodeType')
        if not isinstance(var_count, int):
            raise TemplateError('Invalid var_count. It must be int.')
        self.__type = node_type
        self.__var_count = var_count
        self.__nodes = {}
        self.__goal_keys = set()
        self.__protected_goal_keys = set()
        if name is None:
            self.__name = f'VarNode-{VarNode.__number_of_nodes}'
        else:
            self.__name = name
        VarNode.__number_of_nodes += 1

    def __call__(self, *symbols):
        self.__assert_var_count(symbols)
        return BindedVarNode(self, symbols)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    @classmethod
    def module(cls, var_count, name=None):
        return cls(NodeType.Module, var_count, name)

    @classmethod
    def data(cls, var_count, name=None):
        return cls(NodeType.Data, var_count, name)

    @classmethod
    def goal(cls, var_count, name=None):
        return cls(NodeType.Goal, var_count, name)

    @classmethod
    def protected_goal(cls, var_count, name=None):
        return cls(NodeType.ProtectedGoal, var_count, name)

    @property
    def var_count(self):
        return self.__var_count

    def as_goal(self, *things):
        """
        things が特定の組み合せの場合のみ Goal として扱う
        """
        self.__assert_var_count(things)
        key = self.__to_node_key(things)
        self.__goal_keys.add(key)

    def as_protected_goal(self, *things):
        """
        things が特定の組み合せの場合のみ ProtectedGoal として扱う
        """
        if self.__type is not NodeType.Data:
            pass
        self.__assert_var_count(things)
        key = self.__to_node_key(things)
        self.__protected_goal_keys.add(key)

    def expand(self, network, config, *things):
        self.__assert_var_count(things)
        key = self.__to_node_key(things)
        if key not in self.__nodes:
            node = self.__create_network_node(network, config, things)
            self.__nodes[key] = node
        return self.__nodes[key]

    def __create_network_node(self, network, config, things):
        node_type = self.node_type(things)
        name = config.name(self, things)
        if node_type is NodeType.Module:
            threshold = config.threshold(self, things)
            function = config.function(self, things)
            # logger.info(f'Expand Module: {self.name} -> {name}')
            return network.create_module(threshold, function, name)

        intensity = config.intensity(self, things)
        if node_type is NodeType.Data:
            # logger.info(f'Expand Data: {self.name} -> {name}')
            return network.create_data(intensity, name)
        if node_type is NodeType.Goal:
            # logger.info(f'Expand Goal: {self.name} -> {name}')
            goal_intensity = config.goal_intensity(self, things)
            return network.create_goal(intensity, goal_intensity, name)
        if node_type is NodeType.ProtectedGoal:
            protected_goal_intensity = config.protected_goal_intensity(self, things)
            # logger.info(f'Expand Protected Goal: {self.name} -> {name}')
            return network.create_protected_goal(intensity, protected_goal_intensity, name)
        raise TemplateError('unreachable')

    def node_type(self, things):
        key = self.__to_node_key(things)
        if key in self.__goal_keys:
            return NodeType.Goal
        if key in self.__protected_goal_keys:
            return NodeType.ProtectedGoal
        return self.__type

    def __to_node_key(self, things):
        return '-'.join(map(str, things))

    def __assert_var_count(self, things):
        if len(things) == self.__var_count:
            return
        msg = f'takes {self.__var_count} things but {len(things)} were given'
        raise TemplateError(msg)


class BindedVarNode:
    def __init__(self, var_node, symbols):
        self.__var_node = var_node
        self.__symbols = symbols

    @property
    def var_node(self):
        return self.__var_node

    @property
    def symbols(self):
        return self.__symbols


class Symbol:
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name
