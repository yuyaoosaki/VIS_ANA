from abc import ABCMeta, abstractmethod
from ana.observer import Observable
from enum import Enum, auto
from threading import Thread
from time import time
import re
import logging
from ..VIS import Sensor_data_manager2



logger = logging.getLogger('ana')


class ANAError(Exception):
    pass


CONFLICTOR_LINK_DECAY = 0.4
PREDECESSOR_LINK_DECAY = 0.5
SUCCESSOR_LINK_DECAY = 0.5


def set_conflictor_link_decay(decay):
    global CONFLICTOR_LINK_DECAY
    CONFLICTOR_LINK_DECAY = decay


def set_predecessor_link_decay(decay):
    global PREDECESSOR_LINK_DECAY
    PREDECESSOR_LINK_DECAY = decay


def set_successor_link_decay(decay):
    global SUCCESSOR_LINK_DECAY
    SUCCESSOR_LINK_DECAY = decay


class ANANetwork(Observable):
    def __init__(self):
        super().__init__()
        self.__modules = set()
        self.__statuses = {
            'data': set(),
            'goal': set(),
            'protected_goal': set(),
            '_': set(),
        }
        self.__links = {
            'condition': set(),
            'add': set(),
            'delete': set(),
            '_': set(),
        }
        # self.__execution_strategy = MaxActivationLevelStrategy()
        self.__execution_strategy = WaitModuleStrategy()

    def get_modules(self, query=None, regex=False):
        """
        クエリにマッチするネットワーク中のModule

        Parameters
        ----------
        query: str
            検索文字列

        regex: bool
            True の場合は query を正規表現として扱う

        Returns
        -------
        set of Module
        """
        return self.__execute_query(self.__modules, query, regex)

    def get_statuses(self, query=None, regex=False):
        """
        クエリにマッチするネットワーク中のStatus

        Parameters
        ----------
        query: str
            検索文字列

        regex: bool
            True の場合は query を正規表現として扱う

        Returns
        -------
        set of Status
        """
        matched = set()
        for nodes in self.__statuses.values():
            matched |= self.__execute_query(nodes, query, regex)
        return matched

    def get_goals(self, query=None, regex=False):
        """
        クエリにマッチするネットワーク中のGoal

        Parameters
        ----------
        query: str
            検索文字列

        regex: bool
            True の場合は query を正規表現として扱う

        Returns
        -------
        set of GoalStatus
        """
        return self.__execute_query(self.__statuses['goal'], query, regex)

    def get_protected_goals(self, query=None, regex=False):
        """
        クエリにマッチするネットワーク中のProtected Goal

        Parameters
        ----------
        query: str
            検索文字列

        regex: bool
            True の場合は query を正規表現として扱う

        Returns
        -------
        set of ProtectedGoalStatus
        """
        return self.__execute_query(self.__statuses['protected_goal'], query, regex)

    def get_data(self, query=None, regex=False):
        """
        クエリにマッチするネットワーク中のData

        Parameters
        ----------
        query: str
            検索文字列

        regex: bool
            True の場合は query を正規表現として扱う

        Returns
        -------
        set of DataStatus
        """
        return self.__execute_query(self.__statuses['data'], query, regex)

    def __execute_query(self, nodes, query, regex):
        if query is None:
            return set(nodes)
        if not isinstance(query, str):
            raise ANAError('Invalid query. It must be str.')
        if regex:
            prog = re.compile(query)
            return {node for node in nodes if prog.match(node.name)}
        return {node for node in nodes if node.name == query}

    def get_conditions(self):
        """
        ネットワーク中の全てのCondition

        Returns
        -------
        set of Condition
        """
        return set(self.__links['condition'])

    def get_delete_links(self):
        """
        ネットワーク中の全てのDelete結合

        Returns
        -------
        set of DeleteLink
        """
        return set(self.__links['delete'])

    def get_add_links(self):
        """
        ネットワーク中の全てのAdd結合

        Returns
        -------
        set of AddLink
        """
        return set(self.__links['add'])

    def create_module(self, threshold, function, name=None):
        """
        新しいModuleをネットワークに作成する

        Parameters
        ----------
        threshold: int or float
        function: FunctionBase
        name: str or None

        Returns
        -------
        Module
        """
        node = Module(self, threshold, function, name)
        # logger.info(f'New Module: {node.name}')
        self.__modules.add(node)
        self.notify()
        return node

    def create_data(self, intensity, name=None):
        """
        新しいDataをネットワークに作成する

        Parameters
        ----------
        intensity: int or float
        name: str or None

        Returns
        -------
        DataStatus
        """
        node = DataStatus(self, intensity, name)
        self.add_status(node)
        return node

    def create_goal(self, intensity, goal_intensity=None, name=None):
        """
        新しいGoalをネットワークに作成する

        Parameters
        ----------
        intensity: int or float
        goal_intensity: int or float or None
        name: str or None

        Returns
        -------
        GoalStatus
        """
        if goal_intensity is None:
            goal_intensity = intensity
        node = GoalStatus(self, intensity, goal_intensity, name)
        self.add_status(node)
        return node

    def create_protected_goal(self, intensity, protected_goal_intensity=None, name=None):
        """
        新しいProtected Goalをネットワークに作成する

        Parameters
        ----------
        intensity: int or float
        protected_goal_intensity: int or float or None
        name: str or None

        Returns
        -------
        ProtectedGoalStatus
        """
        if protected_goal_intensity is None:
            protected_goal_intensity = intensity
        node = ProtectedGoalStatus(self, intensity, protected_goal_intensity, name)
        self.add_status(node)
        return node

    def change_as_data(self, status, intensity=None):
        if intensity is None:
            intensity = status.intensity
        return self.__change_status(status, DataStatus, [intensity])

    def change_as_goal(self, status, intensity=None, goal_intensity=None):
        if intensity is None:
            intensity = status.intensity
        if goal_intensity is None:
            goal_intensity = intensity
        return self.__change_status(status, GoalStatus, [intensity, goal_intensity])

    def change_as_protected_goal(self, status, intensity=None, protected_goal_intensity=None):
        if intensity is None:
            intensity = status.intensity
        if protected_goal_intensity is None:
            protected_goal_intensity = intensity
        return self.__change_status(status, ProtectedGoalStatus, [intensity, protected_goal_intensity])

    def __change_status(self, status, status_class, intensities):
        if isinstance(status, status_class):
            raise TypeError()
        if not isinstance(status, Status):
            raise TypeError()
        to_status = status_class(self, *intensities, status.name)

        # もとの Status を削除
        for statuses in self.__statuses.values():
            statuses.discard(status)

        # Link の status を to_status で置換
        for links in self.__links.values():
            for link in links:
                if link.status is status:
                    link.set_status(to_status)

        self.add_status(to_status)
        return to_status

    def add_status(self, status):
        """
        statusをネットワークに追加する

        Parameters
        ----------
        status: Status

        Note
        ----
        標準的なANAから拡張して新しいStatusを定義したい場合は、
        add_statusからネットワークに登録する
        """
        if not isinstance(status, Status):
            raise ANAError('Invalid status. It must be a Status.')
        if isinstance(status, DataStatus):
            # logger.info(f'New Data: {status.name}')
            self.__statuses['data'].add(status)
        elif isinstance(status, GoalStatus):
            # logger.info(f'New Goal: {status.name}')
            self.__statuses['goal'].add(status)
        elif isinstance(status, ProtectedGoalStatus):
            # logger.info(f'New Protected Goal: {status.name}')
            self.__statuses['protected_goal'].add(status)
        else:
            # logger.info(f'New Custom Status: {status.name}')
            self.__statuses['_'].add(status)
        self.notify()

    def create_condition(self, from_node, to_node):
        """
        新しいConditionをネットワークに作成する

        Parameters
        ----------
        from_node: Status
        to_node: Module

        Returns
        -------
        Condition
        """
        link = Condition(from_node, to_node)
        self.add_link(link)
        return link

    def create_delete_link(self, from_node, to_node):
        """
        新しいDelete結合をネットワークに作成する

        Parameters
        ----------
        from_node: Module
        to_node: Status

        Returns
        -------
        DeleteLink
        """
        link = DeleteLink(from_node, to_node)
        self.add_link(link)
        return link

    def create_add_link(self, from_node, to_node):
        """
        新しいAdd結合をネットワークに作成する

        Parameters
        ----------
        from_node: Module
        to_node: Status

        Returns
        -------
        AddLink
        """
        link = AddLink(from_node, to_node)
        self.add_link(link)
        return link

    def add_link(self, link):
        """
        linkをネットワークに追加する

        Parameters
        ----------
        link: Link

        Note
        ----
        標準的なANAから拡張して新しいLinkを定義したい場合は、
        add_linkからネットワークに登録する
        """
        if not isinstance(link, Link):
            raise ANAError('Invalid link. It must be a Link.')
        if isinstance(link, Condition):
            # logger.info(f'New Condition Link: (Module, Status) = ({link.module.name}, {link.status.name})')
            self.__links['condition'].add(link)
        elif isinstance(link, DeleteLink):
            # logger.info(f'New Delete Link: (Module, Status) = ({link.module.name}, {link.status.name})')
            self.__links['delete'].add(link)
        elif isinstance(link, AddLink):
            # logger.info(f'New Add Link:  (Module, Status) = ({link.module.name}, {link.status.name})')
            self.__links['add'].add(link)
        else:
            # logger.info(f'New Custom Link: (Module, Status) = ({link.module.name}, {link.status.name})')
            self.__links['_'].add(link)
        self.notify()

    def decay_activation(self):
        """
        Decay functionを実行する
        """
        # logger.info('[ Decay Function ]')
        DECAY_RATE = 0.99
        for module in self.__modules:
            module.decay_activation(DECAY_RATE)
        self.notify()

    def execute(self, t):
        """
        活性済かつ FunctionExecutionStrategy に選択されたModule を実行
        """
        candidates = []
        for module in self.__modules:
            if module.is_active(t):
                candidates.append(module)
        # for candidate in candidates:
        #     print('[', candidate.name, candidate.activation_level, ']')
        self.__execution_strategy.execute(candidates, t)
        self.notify()

    def __get_all_links(self):
        for links in self.__links.values():
            for link in links:
                yield link

    def propagate_externally(self):
        """
        External loopの活性伝播を実行する
        """
        # logger.info('[ External Loop ]')
        for link in self.__get_all_links():
            link.propagate_externally()
        for link in self.__get_all_links():
            link.reflect_propagation()
        self.notify()

    def propagate_internally(self):
        """
        Internal loopの活性伝播を実行する
        """
        # logger.info('[ Internal Loop ]')
        for link in self.__get_all_links():
            link.propagate_internally()
        for link in self.__get_all_links():
            link.reflect_propagation()
        self.notify()

    def prev_nodes_of(self, node):
        """
        Parameters
        ----------
        node: Node

        Returns
        -------
        set of Node
            nodeの前にある全てのNode
        """
        nodes = set()
        for link in self.__get_all_links():
            if node is link.to_node:
                nodes.add(link.from_node)
        return nodes

    def next_nodes_of(self, node):
        """
        Parameters
        ----------
        node: Node

        Returns
        -------
        set of Node
            nodeの次にある全てのNode
        """
        nodes = set()
        for link in self.__get_all_links():
            if node is link.from_node:
                nodes.add(link.to_node)
        return nodes

    def __gather_from_node(self, to_node, links):
        nodes = set()
        for link in links:
            if to_node is link.to_node:
                nodes.add(link.from_node)
        return nodes

    def __gather_to_node(self, from_node, links):
        nodes = set()
        for link in links:
            if from_node is link.from_node:
                nodes.add(link.to_node)
        return nodes

    def condition_statuses_of(self, module):
        """
        Parameters
        ----------
        module: Module

        Returns
        -------
        set of Status
            moduleのCondition list
        """
        return self.__gather_from_node(module, self.get_conditions())

    def condition_modules_of(self, status):
        """
        Parameters
        ----------
        status: Status

        Returns
        -------
        set of Module
            Condition listにstatusを含む全てのModule
        """
        return self.__gather_to_node(status, self.get_conditions())

    def add_statuses_of(self, module):
        """
        Parameters
        ----------
        module: Module

        Returns
        -------
        set of Status
            moduleのAdd list
        """
        return self.__gather_to_node(module, self.get_add_links())

    def add_modules_of(self, status):
        """
        Parameters
        ----------
        status: Status

        Returns
        -------
        set of Module
            Add listにstatusを含む全てのModule
        """
        return self.__gather_from_node(status, self.get_add_links())

    def delete_statuses_of(self, module):
        """
        Parameters
        ----------
        module: Module

        Returns
        -------
        set of Status
            moduleのDelete list
        """
        return self.__gather_to_node(module, self.get_delete_links())

    def delete_modules_of(self, status):
        """
        Parameters
        ----------
        status: Status

        Returns
        -------
        set of Module
            Delete listにstatusを含む全てのModule
        """
        return self.__gather_from_node(status, self.get_delete_links())


class Node(metaclass=ABCMeta):
    __number_of_nodes = 0

    def __init__(self, network, name=None):
        if not isinstance(network, ANANetwork):
            raise ANAError('Invalid network. It must be a ANANetwork.')
        self._network = network
        if name is None:
            self.__name = f'Node-{Node.__number_of_nodes}'
        else:
            self.__name = name
        Node.__number_of_nodes += 1

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    @property
    def prev_nodes(self):
        """set of Node: 前にあるNode"""
        return self._network.prev_nodes_of(self)

    @property
    def next_nodes(self):
        """set of Node: 次にあるNode"""
        return self._network.next_nodes_of(self)

    @abstractmethod
    def is_active(self):
        """
        Returns
        -------
        bool
            活性状態
        """
        

class FunctionBase(metaclass=ABCMeta):
    """
    Moduleが発火したときに実行すべき機能
    """

    @abstractmethod
    def execute(self, module):
        """
        Parameters
        ----------
        module: Module

        Note
        ----
        Moduleが発火したときに実行される
        """


class ModuleFunction(FunctionBase):
    def __init__(self, sensor_data_manager):
        self.sensor_data_manager = sensor_data_manager
        self.t = None

    def execute(self, module):
        self._function(module)
        module.reset_activation()

    @abstractmethod
    def _function(self, module):
        pass


class ParallelModuleFunction(FunctionBase):
    def execute(self, module):
        def target(module):
            self._function(module)
            module.reset_activation()
        Thread(target=target, daemon=True).start()

    @abstractmethod
    def _function(self, module):
        pass


class FunctionExecutionStrategy(metaclass=ABCMeta):
    """
    Module の発火方法を決定する
    """

    def __init__(self):
        self.__executing_module = None

    # def execute(self, candidate_modules, t):
    #     module = self._select_module(candidate_modules, t)
    #     if module is None:
    #         self.__executing_module = None

    #     elif self.__executing_module is not module:
    #         print('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', module.name)
    #         module.execute(t)
    #         self.__executing_module = module
    #     else:
    #         print('×××××××××××××', module.name)

    def execute(self, candidate_modules, t):
        module = self._select_module(candidate_modules, t)
        if module is not None:
            module.execute(t)
            print('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', module.name)


    @abstractmethod
    def _select_module(self, candidate_modules):
        """
        自らの戦略に従って実行すべき Module を選択する

        Returns
        -------
        Module or None
            実行する Module を返す。実行すべき Module がない場合は None を返す。
        """


class MaxActivationLevelStrategy(FunctionExecutionStrategy):
    """activation_level が最大の Module を実行"""

    def _select_module(self, candidate_modules):
        max_activation_level = -float('inf')
        max_module = None
        for module in candidate_modules:
            if module.activation_level > max_activation_level:
                max_module = module
                max_activation_level = max_module.activation_level
        return max_module


class StackStrategy(FunctionExecutionStrategy):
    """直近で activation_level が閾値に達した Module を実行"""

    def __init__(self):
        super().__init__()
        self.__last_stack = []

    def _select_module(self, candidate_modules):
        modules = sorted(candidate_modules, key=lambda m: m.activation_level)
        stack = []
        for module in self.__last_stack:
            if module in modules:
                stack.append(module)
        for module in modules:
            if module not in stack:
                stack.append(module)
        self.__last_stack = stack
        if len(stack) == 0:
            return None
        return stack[-1]


class StackReverseStrategy(FunctionExecutionStrategy):
    """直近で activation_level が閾値に達した Module 以外を実行"""

    def __init__(self):
        super().__init__()
        self.__last_stack = []

    def _select_module(self, candidate_modules):
        modules = sorted(candidate_modules, reverse=True, key=lambda m: m.activation_level)
        self.__last_stack.sort(reverse=True, key=lambda m: m.activation_level)
        stack = []
        for module in self.__last_stack:
            if module in modules:
                stack.append(module)
        for module in modules:
            if module not in stack:
                stack.append(module)
        self.__last_stack = stack
        if len(stack) == 0:
            return None
        return stack[0]


class WaitModuleStrategy(FunctionExecutionStrategy):
    """activation_level が閾値を超えてからの経過時間が長い module を優先的に実行"""
    def __init__(self):
        self.wait_over = 100

    def _select_module(self, candidate_modules, t):
        max_activation_level = -float('inf')
        max_module = None
        for module in candidate_modules:
            wait_value = (t - module._over_time) * self.wait_over
            print(f'[[{module.name} = {int(module.activation_level)} + {int(wait_value)} = {int(module.activation_level + wait_value)}]]')
            if module.activation_level + wait_value > max_activation_level:
                max_module = module
                max_activation_level = max_module.activation_level + wait_value
        return max_module


class Module(Node):
    """
    Parameters
    ----------
    network: ANANetwork
    threshold: int of float
        activation levelの閾値
    function: FunctionBase
        Moduleに紐付く機能
    name: str or None
        Module の名前
    """

    def __init__(self, network, threshold, function, name=None):
        super().__init__(network, name)
        if not isinstance(threshold, (int, float)):
            raise ANAError('Invalid threshold. It must be int or float.')
        if not isinstance(function, FunctionBase):
            raise ANAError('Invalid function. It must be a FunctionBase.')
        self.__threshold = threshold
        self.__function = function
        self.__activation_level = 0
        self._over_time = None

    @property
    def activation_level(self):
        """int of float: activation level"""
        return self.__activation_level

    @property
    def condition_statuses(self):
        """set of Status: Condition list"""
        return self._network.condition_statuses_of(self)

    @property
    def add_statuses(self):
        """set of Status: Add list"""
        return self._network.add_statuses_of(self)

    @property
    def delete_statuses(self):
        """set of Status: Delete list"""
        return self._network.delete_statuses_of(self)

    def are_all_condition_nodes_active(self):
        """
        Returns
        -------
        bool
            Condition listに含まれる全てのStatusのFlagが立っているか
        """
        for status in self.condition_statuses:
            if not status.is_active():
                return False
        return True

    def execute(self, t):
        """機能を実行する"""
        if not self.is_active(t):
            raise ANAError('activateでないのに実行された')
        self.__function.execute(self, t)

    def reset_activation(self):
        self.__activation_level = 0

    def decay_activation(self, rate):
        """
        activation levelを減衰させる

        Parameters
        ----------
        rate: float
            減衰率
        """
        if not isinstance(rate, float):
            raise ANAError('Invalid rate. It must be float.')
        self.__activation_level *= rate
        self.__clamp_activation_level()

    def propagate_activation(self, intensity):
        """
        activation levelを増減させる

        Parameters
        ----------
        intensity: int or float
            加算するactivation levelの値
        """
        if not isinstance(intensity, (int, float)):
            raise ANAError('Invalid intensity. It must be int or float.')
        self.__activation_level += intensity
        self.__clamp_activation_level()

    def __clamp_activation_level(self):
        self.__activation_level = max(self.__activation_level, 0)

    # def is_active(self):
    #     if self.activation_level < self.__threshold:
    #         return False
    #     return self.are_all_condition_nodes_active()

    def is_active(self, t):
        if self.activation_level < self.__threshold:
            self._over_time = None
            return False
        elif self._over_time is None:
            self._over_time = t
        return self.are_all_condition_nodes_active()


class Status(Node):
    def __init__(self, network, intensity, name=None):
        if not isinstance(intensity, (int, float)):
            raise ANAError('Invalid intensity. It must be int or float.')
        super().__init__(network, name)
        self.__status = False
        self.__intensity = intensity
        self.__activated_time = None
        self.__object_position = None
        self.sensor_data_manager = None

    @property
    def intensity(self):
        """int or float: intensity"""
        return self.__intensity

    @property
    def condition_modules(self):
        """set of Module: Condition listにselfを含むModule"""
        return self._network.condition_modules_of(self)

    @property
    def add_modules(self):
        """set of Module: Add listにselfを含むModule"""
        return self._network.add_modules_of(self)

    @property
    def delete_modules(self):
        """set of Module: Delete listにselfを含むModule"""
        return self._network.delete_modules_of(self)

    def activate(self, t):
        """Flagを立てる"""
        self.__status = True
        self.__activated_time = t

    def deactivate(self):
        """Flagを下す"""
        self.__status = False
        self.__object_position = None
        self.sensor_data_manager = None

    def set_object_position(self, e2c_data, bb):
        self.__object_position = (e2c_data, bb)
        self.sensor_data_manager = Sensor_data_manager2()

    def is_active(self):
        return self.__status

    @property
    def activated_time(self):
        return self.__activated_time

    @property
    def object_position(self):
        return self.__object_position


class DataStatus(Status):
    pass


class GoalStatus(Status):
    """エージェントの目的"""

    def __init__(self, network, intensity, goal_intensity, name=None):
        if not isinstance(goal_intensity, (int, float)):
            raise ANAError('Invalid goal_intensity. It must be int or float.')
        super().__init__(network, intensity, name)
        self.__goal_intensity = goal_intensity

    @property
    def goal_intensity(self):
        return self.__goal_intensity


class ProtectedGoalStatus(Status):
    """エージェントの達成したくない目的"""

    def __init__(self, network, intensity, protected_goal_intensity, name=None):
        if not isinstance(protected_goal_intensity, (int, float)):
            raise ANAError('Invalid protected_goal_intensity. It must be int or float.')
        super().__init__(network, intensity, name)
        self.__protected_goal_intensity = protected_goal_intensity

    @property
    def protected_goal_intensity(self):
        return self.__protected_goal_intensity


class Link(metaclass=ABCMeta):
    class Direction(Enum):
        From = auto()
        To = auto()
    _MODULE = Direction.From

    def __init__(self, from_node, to_node):
        FROM_NODE_CLASS = Module if self._MODULE is Link.Direction.From else Status
        TO_NODE_CLASS = Status if self._MODULE is Link.Direction.From else Module
        if not isinstance(from_node, FROM_NODE_CLASS):
            class_name = FROM_NODE_CLASS.__name__
            raise ANAError(f'Invalid from_node. It must be {class_name}.')
        if not isinstance(to_node, TO_NODE_CLASS):
            class_name = TO_NODE_CLASS.__name__
            raise ANAError(f'Invalid to_node. It must be {class_name}.')
        self.__from_node = from_node
        self.__to_node = to_node
        self.__propagation_store = 0

    def set_status(self, status):
        if not isinstance(status, Status):
            raise TypeError()
        if self._MODULE is Link.Direction.From:
            self.__to_node = status
        else:
            self.__from_node = status

    def set_module(self, module):
        if not isinstance(module, Module):
            raise TypeError()
        if self._MODULE is Link.Direction.From:
            self.__from_node = module
        else:
            self.__to_node = module

    @property
    def status(self):
        if self._MODULE is Link.Direction.From:
            return self.to_node
        return self.from_node

    @property
    def module(self):
        if self._MODULE is Link.Direction.From:
            return self.from_node
        return self.to_node

    @property
    def from_node(self):
        return self.__from_node

    @property
    def to_node(self):
        return self.__to_node

    @abstractmethod
    def propagate_externally(self):
        """
        External loopで伝播すべき値を計算する。
        External loopで一度だけ実行される。

        Note
        ----
        計算した伝播値は_store_propagationを通してmoduleに反映させる
        """

    @abstractmethod
    def propagate_internally(self):
        """
        Internal loopで伝播すべき値を計算する。
        Internal loopで一度だけ実行される。

        Note
        ----
        計算した伝播値は_store_propagationを通してmoduleに反映させる
        """

    def reflect_propagation(self):
        """
        _store_propagationで登録された活性伝播値をmoduleに反映させる
        """
        self.module.propagate_activation(self.__propagation_store)
        self.__propagation_store = 0

    def _store_propagation(self, intensity):
        """
        一括で実行すべき活性伝播を登録する。

        property
        --------
        intensity: int or float
            活性伝播値
        """
        self.__propagation_store += intensity


class Condition(Link):
    """
    Moduleが活性化するためにはStatusのFlagがTrueであることが必要
    """

    _MODULE = Link.Direction.To

    def propagate_externally(self):
        status = self.status
        if status.is_active():
            intensity = status.intensity
            # logger.info(f'Spreading Activation (from active status): {status.name} -> {self.module.name} ({intensity})')
            self._store_propagation(intensity)

    def propagate_internally(self):
        status = self.status
        for successor_module in status.add_modules:
            if successor_module.are_all_condition_nodes_active():
                intensity = SUCCESSOR_LINK_DECAY * successor_module.activation_level
                # logger.info(f'Spreading Activation (from successor module): {successor_module.name} -> {self.module.name} ({intensity})')
                self._store_propagation(intensity)


class AddLink(Link):
    """
    Moduleを実行するとStatusがTrueになる
    """

    def propagate_externally(self):
        status = self.status
        if isinstance(status, GoalStatus):
            intensity = status.goal_intensity
            # logger.info(f'Spreading Activation (from goal): {status.name} -> {self.module.name} ({intensity})')
            self._store_propagation(intensity)

    def propagate_internally(self):
        status = self.status
        for predecessor_module in status.condition_modules:
            if not predecessor_module.are_all_condition_nodes_active():
                intensity = PREDECESSOR_LINK_DECAY * predecessor_module.activation_level
                # logger.info(f'Spreading Activation (from predecessor module): {predecessor_module.name} -> {self.module.name} ({intensity})')
                self._store_propagation(intensity)
        for conflictor_module in status.delete_modules:
            intensity = -CONFLICTOR_LINK_DECAY * conflictor_module.activation_level
            # logger.info(f'Spreading Activation (from conflictor module): {conflictor_module.name} -> {self.module.name} ({intensity})')
            self._store_propagation(intensity)


class DeleteLink(Link):
    """
    Moduleを実行するとStatusがFalseになる
    """

    def propagate_externally(self):
        status = self.status
        if isinstance(status, ProtectedGoalStatus):
            intensity = -status.protected_goal_intensity
            # logger.info(f'Spreading Activation (from protected goal): {status.name} -> {self.module.name} ({intensity})')
            self._store_propagation(intensity)

    def propagate_internally(self):
        pass
