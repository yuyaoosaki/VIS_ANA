from abc import ABCMeta, abstractmethod


class Observable:
    def __init__(self):
        self.__observers = []
        self.__holding = False
        self.__notifying = True

    def add_observer(self, observer):
        if not isinstance(observer, Observer):
            raise TypeError('Invalid observer. It must be Observer.')
        self.__observers.append(observer)

    def notify(self):
        if not self.__notifying:
            self.__holding = True
            return
        for observer in self.__observers:
            observer.update(self)

    def stop_notifying(self):
        self.__notifying = False

    def start_notifying(self):
        self.__notifying = True
        if self.__holding:
            self.notify()
            self.__holding = False


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, observable):
        pass
