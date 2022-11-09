from ana.observer import Observer
from threading import Thread, Lock
import eel
import logging

logger = logging.getLogger('ana')


class Batcher():
    def __init__(self):
        self.__items = []
        self.__lock = Lock()

    def batch(self, func, *args):
        item = BatchItem(func, args)
        if self.__items:
            self.__items[-1].set_next_batch_item(item)
        self.__items.append(item)
        return self

    def end_batch(self):
        self.__lock.acquire()
        if self.__items:
            self.__items[0].execute()
        self.__items = []
        self.__lock.release()


class BatchItem:
    def __init__(self, func, args):
        self.__func = func
        self.__args = args
        self.__next = None

    def set_next_batch_item(self, batch_item):
        self.__next = batch_item

    def execute(self):
        if self.__next is None:
            self.__func(*self.__args)
        else:
            self.__func(*self.__args)(lambda _: self.__next.execute())


class NetworkView(Observer):
    __ready_eel = False
    __instances = []

    def __init__(self):
        self.__network = None
        NetworkView.__instances.append(self)
        Thread(target=self.__start, daemon=True).start()

    def __start(self):
        # logger.info('Start')
        eel.init('ana/view/web')
        eel.start('main.html', close_callback=self.__close_callback)

    def __close_callback(self, route, websockets):
        # logger.info('Close')
        NetworkView.__ready_eel = False

    @staticmethod
    @eel.expose
    def ready():
        # logger.info('Ready')
        NetworkView.__ready_eel = True
        for view in NetworkView.__instances:
            view.update_eel()

    def update(self, network=None):
        self.__network = network
        self.update_eel()

    def update_eel(self):
        if not NetworkView.__ready_eel or self.__network is None:
            return

        def to_node_data(nodes):
            return [{'id': node.name, 'active': node.is_active()} for node in nodes]

        def to_module_data(modules):
            return [{'id': module.name, 'activationLevel': module.activation_level} for module in modules]

        def to_id_pair(links):
            pairs = []
            for link in links:
                pairs.append([link.from_node.name, link.to_node.name])
            return pairs

        # logger.info('Update')
        Batcher().batch(eel.cyStartBatch)\
                 .batch(eel.setModules, to_module_data(self.__network.get_modules()))\
                 .batch(eel.setData, to_node_data(self.__network.get_data()))\
                 .batch(eel.setGoals, to_node_data(self.__network.get_goals()))\
                 .batch(eel.setProtectedGoals, to_node_data(self.__network.get_protected_goals()))\
                 .batch(eel.setConditionLinks, to_id_pair(self.__network.get_conditions()))\
                 .batch(eel.setAddLinks, to_id_pair(self.__network.get_add_links()))\
                 .batch(eel.setDeleteLinks, to_id_pair(self.__network.get_delete_links()))\
                 .batch(eel.cyEndBatch)\
                 .batch(eel.cyLayout)\
                 .end_batch()
