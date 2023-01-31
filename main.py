from dataclasses import dataclass
import networkx as nx
import numpy as np
from collections import deque
from typing import Callable, Iterable


@dataclass(frozen=True)  # (можно просто tuple или как удобнее)
class Task:
    calc_size: float  # размер задачи для выполнения
    transfer_weight: int  # размер задачи для передачи данных в байтах
    transfer_weight_return: int  # размер результата задачи для передачи данных
    customer_id: int  # id of a customer of the task


class Node:
    # SUGGESTION FOR ALEXANDER:
    # I think that sources and destinations of each node must be stored in the `Net` class
    """
        This class simulates behavior of ...
    """

    def __init__(self,
                 x_0: float,  # position at the begining of a simualtion
                 y_0: float,
                 power: float,  # computing power
                 # f : (time:float) -> (x:float, y:float)
                 way_equation: Callable,
                 ):
        self.x = x_0
        self.y = y_0
        self.power = power
        self.isActive = True  # True if the node is capable of interacting with others
        self.isCalculating = False  # True if the node is computing some task
        self.isTransfering = False  # True if the node is transfering some data
        self.way_equation = way_equation
        self.route = []
        self.tasks = deque()  # queue of tasks for node to compute
        self.current_progress = 0

    def __update_state(self, ):
        self.isCalculating = bool(self.tasks)

    def forget_tasks(self, to_forget: Iterable):
        new_tasks = deque()
        for task in self.tasks:
            if task.customer_id not in to_forget:
                new_tasks.append(task)
        self.tasks = new_tasks

    def calc(self, timedelta: float) -> list[Task]:
        """
            This method simulates process of node computing the tasks. 
            timedelta: float - computing time
        """
        self.current_progress += self.power * timedelta
        if self.current_progress - self.tasks[0].calc_size >= 0:
            self.current_progress -= self.tasks[0].calc_size
            finished_task = self.tasks.popleft()
        self.__update_state()
        # eventually checks if the tasks are empty,
        # WE HAVE JUST 20ms TIMESTEP, SO I DECIDED THAT AT ONE TIMESTEP ONLY ONE
        # TASK CAN BE FINISHED (I PRESUMED THAT TASKS ARE NOT SO SMALL)
        # while self.tasks and self.current_progress >= self.tasks[0].calc_size:
        #     self.current_progress -= self.tasks[0].calc_size
        #     finished_tasks.append(self.tasks.popleft())
        return finished_task

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.__update_state()

    def wake(self, ):
        self.isActive = True

    def sleep(self, ):
        self.isActive = False

    def move(self, t):
        x, y = self.way_equation(t)
        self.x = x
        self.y = y

# расстояние между узлами


def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class Net:
    def __init__(
        self,
        # bandwidth_formula: max_bandwidth -> (f: distance -> bandwidth)
        bandwidth_formula: callable,
        nodes: dict()  # dict {node_id : Node}
    ):
        self.nodes: dict() = nodes
        self.G = nx.Graph()  # может иметь произвольное количество компонент связности, должен динамически меняться в зависимотсти от положение узлов
        # 100 мб/c. Меняет в зависимости от растояния по нелинейным формулам
        self.max_bandwidth = 100
        # максимальное расстояние на котором поддерживается свзять 30м. Если расстояние больше, то связь разорвана
        self.max_distance = 30
        # считаем силу сигнала в зависимости от расстояния по этой формуле. должна учитывать max_bandwidth
        self.bandwidth_formula = bandwidth_formula(self.max_distance)

        # we need variable to store state of the network.
        # this variable must store information about following:
        # 1. customers and performers of each task
        # 2. sources and destinations of each node

        self.task_customers = {}  # {task_id : node_id (node is a customer)}
        # {node_id: list of ids of performers}
        self.performers = Net.__init_performers(self.nodes.keys())
        # {node_id: destination of data transmission} # if destination is 0, then node doesn't transimit data
        self.destinations = Net.__init_destinations(self.nodes.keys())

    def __init_destinations(node_ids):
        d = dict()
        for id in node_ids:
            d[id] = 0
        return d

    def __init_performers(node_ids):
        d = dict()
        for id in node_ids:
            d[id] = []
        return d

    def move(self, t):
        """
            Call this function to move the nodes according to their 
            way_equation formulas.
        """
        for node in self.nodes:
            node.move(t)

    # update graph's components
    def __update_components(self,):
        # при переопределении ребра, информация о прошлом ребре стирается
        for i in range(len(self.nodes)):
            for j in range(i, len(self.nodes)):
                x = self.nodes[i]
                y = self.nodes[j]
                d = dist(x, y)
                if d < self.max_distance:
                    self.G.add_edge(x.node_id, y.node_id,
                                    weight=self.bandwidth_formula(d))
                else:
                    self.G.remove_edge(x.node_id, y.node_id)

    def __check_connection(self, id_1, id_2):
        # !!!!!!
        return True

    def add_task(self, customer_id, task: Task):
        self.nodes[customer_id].tasks.append(task)

    def update(self, timestep):
        to_be_sent_back = []

        self.move(timestep)
        self.__update_components()
        for node_id in self.nodes.keys():

            self.nodes[node_id].move()
            if self.destinations[node_id]:
                is_connected = self.check_connection(
                    node_id, self.destinations[node_id])
                if not is_connected:
                    self.destinations[node_id] = 0

            # забываем о тех задачах, которые были выданы узлами, покинувшими компоненту связности узла
            lost_connections = {}
            for task in self.nodes[node_id].tasks:
                customer_id = task.customer_id
                if self.__check_connection(node_id, customer_id):
                    lost_connections.add(customer_id)
            self.nodes[node_id].forget_tasks(lost_connections)

            # collect data to be sent back
            finished_task = self.node.calc()
            to_be_sent_back.append(finished_task)
        
        return to_be_sent_back
            # 1) проверить что передача данных все еще идет и все в порядке (нет разрыва сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
            # 2) проверить что вычисления все еще имеют смысл (тот для кого мы вычисляем все еще в сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
            # 3) если первые два условия не выполняются, то можно написать алгоритмы опитмизации, 
            #    чтобы по возращению устрйоств в сеть они продоолжали выполнять прерванное (НЕ РЕАЛИЗОВАНО)

    def schedule():
        pass
        # for node in self.nodes:
        #     for task ... 
        #     node_dest = scheduler(node, node.tasks[-1])
        #     # вычисляем путь до того, на ком скедулить
        #     rout = net.shortest_path(node, node_dest)
        #     # remove_task(node, -1)
        #     node.rout = rout

    def shortest_path(self,from_, to_):
        self.__update_components()
        D=nx.DiGraph()
        for i in self.G.nodes:
            for j in self.G.nodes:
                if (i,j) in self.G.edges and not (self.nodes[i].isTransfering or self.nodes[j].isTransfering):
                    D.add_edge(i,j,weight=self.G.edges[i,j]['weight'])
                    D.add_edge(j,i,weight=self.G.edges[i,j]['weight'])
                    
        for i in nx.weakly_connected_components(D):
            if from_ and to_ in i:
                cost=dict.fromkeys(D.nodes,-1)
                cost[from_]=float("Inf")
                vertexes=dict.fromkeys(D.nodes)
                for i in range(D.number_of_nodes()- 1): 
                    for u, v, w in D.edges(data=True): 
                        if cost[u] != -1 and min(w['weight'],cost[u]) > cost[v]:
                            cost[v] = min(w['weight'],cost[u])
                            vertexes[v]=u
                route=[to_]
                r=to_
                while r != from_:
                    r=vertexes[r]
                    route.insert(0,r)
                return route
            
        return [from_]


class Simulation:
    def __init__(self,):

        # соколько щагов проссимулировать. 1 шаг == 20мс (автоматически дает нам в среднем залержку в 10мс) (180.000 == 1 час)
        self.steps: int
        self.net: Net
        self.scentrio: Scenario
        self.schedule_interval = ...  # как часто делаем скедулинг в ms

    # перемещения узлом и многое другое можно визуализировать через анимации. То есть в процессе строить анимацию, а в конце записать ее в файл .gif
    def visualization(self,):
        # не мы
        pass

    def run(self,):
        for timestep in range(self.steps):
            self.net.update(timestep)
            if timestep % self.schedule_interval == 0:
                self.net.schedule()

# 1. Сценарии. Разобрать с генератором задач
# 2. Переменная хранения состояния сети, интерфейс для использования в Simulator (описан в init класса Net)
# 3. Написать примитивный скедулинг задач. (из описания ниже)
# 4. дописать функцию self.run()
# 5. Визуализация
# 6. Вывод нужной информации в какой-то файлик. Определится с измеряемыми метриками симуляции.
# 7. Подумать над гипотезой о том, что распр. вычисления не эффективны. Подумать как ее проверить (мб александру)


@dataclass(frozen=True)
class Scenario:
    # описание сценария. Передается в конструктор симуляции. Можно написать как вот такой объект, чтобы было наглядно,
    # либо можно написать каждое поле отдельной переменной и передавать в разобранном виде в конструктор
    # data:
    # data:
    # data:
    def __init__(self, ):
        self.way_eqs: list[callable] = ...
        # две возможности задания тасков
        # 1) весь список задач изначально известен. новые задачи не появляются в процессе симуляции
        # 2) задачи генерируются каким-то генератором задач. соотвественно, доп задача напиать генератор задач (от time)
        # т.е появление задач привязано к какому-то времени
        self.tasks_list = ...  # or task generator in 2 case
        self.config_devices = ...


"""
помним про некоторые вещи
1) нужно будет сделать baseline - скедулер выбирает в качестве исполнителя исходный узел, нет передачи данных по сети, нет маршрутов
2) в процессе выполнения задач нужно иметь некоторый счетчик прогресса выполнения задач и передачи данных, чтобы было понятно, что задача выполнена
3) выполненная задача это еще не конец - необходимо вернуть результат исходному устройству
4) алгоритм выбора исполненителя:
    в компоненте связности каждому ребру присваиваем transfer_weight / bandwidth
    запускаем беллмана форда для source -> получаем расстояния до всех других вершин
    добавляем к каждму расстоянию calc_size / power
    выбираем наименьшее число из полученных - это тот кто выполнит нашу задачу быстрее всех с учетом передачи данных

    так как нам еще нужно вернуть результат в source, то на первом этапе прибавляем еще аналогичное слагаемое для transfer_weight_return

5) по хорошему нам нужен алгоритм предсказания, но придумать его - не слишком очевидная затея, так что пока без него
6) не стройте вырожденные сценарии. Размер данных для передачи по сети не должен быть слишком большой
7) количество узлов 3-15
"""
