from dataclasses import dataclass
import networkx as nx
import numpy as np
from collections import deque

@dataclass(frozen=True)  # (можно просто tuple или как удобнее)
class Task:
    calc_size: int  # размер задачи для выполнения
    transfer_weight: int  # размер задачи для передачи данных
    transfer_weight_return: int  # размер результата задачи для передачи данных

class Node:
    # SUGGESTION FOR ALEXANDER:
    # I think that sources and destinations of each node must be stored in the `Net` class
    """
        This class simulates behavior of ...
    """
    def __init__(self, 
        node_id: int, 
        x_0: float, # position at the begining of a simualtion  
        y_0: float, 
        power: float, # computing power
        way_equation: function, # f : (time:float) -> (x:float, y:float)
    ):
        self.node_id = node_id
        self.x = x_0
        self.y = y_0
        self.power = power
        self.isActive = True # True if the node is capable of interacting with others
        self.isCalculating = False # True if the node is computing some task 
        self.isTransfering = False # True if the node is transfering some data
        self.way_equation = way_equation
        self.route = [] 
        self.tasks = deque() # queue of tasks for node to compute
        self.current_progress = 0

    def __update_state(self, ):
        self.isCalculating = bool(self.tasks)

    def calc(self, timedelta: float):
        """
            This method simulates process of node computing the tasks. 
            timedelta: float - computing time
        """
        self.current_progress += self.power * timedelta
        while self.tasks and self.current_progress >= self.tasks[0].calc_size:
            self.current_progress -= self.tasks[0].calc_size
            self.tasks.popleft()
        self.__update_state()
        # eventually checks if the tasks are empty,
        # if so, set isCalculating=False (=True otherwise) 
        
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
        bandwidth_formula : function # bandwidth_formula: max_bandwidth -> (f: distance -> bandwidth) 
    ):
        self.nodes: dict()  # набор узлов в формате node_id : Node 
        self.G = nx.Graph()  # может иметь произвольное количество компонент связности, должен динамически меняться в зависимотсти от положение узлов
        self.max_bandwidth = 100  # 100 мб/c. Меняет в зависимости от растояния по нелинейным формулам
        self.max_distance = 30  # максимальное расстояние на котором поддерживается свзять 30м. Если расстояние больше, то связь разорвана
        self.bandwidth_formula = bandwidth_formula(self.max_distance) # считаем силу сигнала в зависимости от расстояния по этой формуле. должна учитывать max_bandwidth

    def move(self, t):
        """
            Call this function to move the nodes according to their 
            way_equation formulas.
        """
        for node in self.nodes:
            node.move(t)

    # обновляем компоненты связности 
    def update_components(self,):
        # при переопределении ребра, информация о прошлом ребре стирается
        for i in range(len(self.nodes)):
            for j in range(i,len(self.nodes)):
                x = self.nodes[i]
                y = self.nodes[j]
                d = dist(x,y)
                if d < self.max_distance:
                    self.G.add_edge(x.node_id, y.node_id, weight=self.bandwidth_formula(d))
                else: 
                    self.G.remove_edge(x.node_id,y.node_id)

    def remove_tasks(node, source):
        new_tasks = []
        for i in range(len(node.tasks)):
            if node.tasks[i].chief != source:
                new_tasks.append(node.tasks[i])
        node.tasks = new_tasks

    def update(self, timestep):
        self.move(timestep)
        self.update_components()
        for node in self.nodes:
            node.calc()
            if node.source:
                if node.source.node_id not in nx.node_connected_component(self.G, node.node_id):
                    Net.remove_tasks(node, node.source)
                    node.source = None
            elif node.destination:
                if node.destination.node_id not in nx.node_connected_component(self.G, node.node_id):
                    Net.remove_tasks(node.destination, node)
                    node.destination = None
                # is_same_component()
                # проверить что передача данных все еще идет и все в порядке (нет разрыва сети). Иначе - обработать ситуацию
            if (node.isCalculating):
                pass
                # проверить что вычисления все еще имеют смысл (тот для кого мы вычисляем все еще в сети). Иначе - обработать ситуацию

                # если первые два условия не выполняются, то можно написать алгоритмы опитмизации, чтобы по возращению устрйоств в сеть они продоолжали выполнять прерванное
    def schedule():
        pass
        # if (node.tasks):  # если есть что скедулить
        #     # вычисляем на ком скедулить
        #     node_dest = scheduler(node, node.tasks[-1])
        #     # вычисляем путь до того, на ком скедулить
        #     rout = net.shortest_path(node, node_dest)
        #     # remove_task(node, -1)
        #     node.rout = rout

    def shortest_path(from_, to_):
        
        pass


class Simulation:
    def __init__(self,):

        # соколько щагов проссимулировать. 1 шаг == 20мс (автоматически дает нам в среднем залержку в 10мс) (180.000 == 1 час)
        self.steps: int
        self.net: Net

    # перемещения узлом и многое другое можно визуализировать через анимации. То есть в процессе строить анимацию, а в конце записать ее в файл .gif
    def visualization(self,): 
        # не мы
        pass

    def run(self,):
        schedule_interval = 10 # как часто делаем скедулинг
        for timestep in range(self.steps):
            self.net.update(timestep)
            if timestep % schedule_interval:
                self.net.schedule()





@dataclass(frozen=True)
class Scenario:
    # описание сценария. Передается в конструктор симуляции. Можно написать как вот такой объект, чтобы было наглядно,
    # либо можно написать каждое поле отдельной переменной и передавать в разобранном виде в конструктор
    # data:
    # data:
    # data:
    pass

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
