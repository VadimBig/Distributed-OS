from dataclasses import dataclass
import networkx as nx
import numpy as np
from collections import deque
from brownian import brownian, constraints_to_brownian, eq_circle, eq_partline, eq_sin_or_cos
import json
import math

@dataclass(frozen=True)  # (можно просто tuple или как удобнее)
class Task:
    calc_size: int  # размер задачи для выполнения
    transfer_weight: int  # размер задачи для передачи данных
    transfer_weight_return: int  # размер результата задачи для передачи данных
    time_to_create: float # время появления задачи

class Node:
    # SUGGESTION FOR ALEXANDER:
    # I think that sources and destinations of each node must be stored in the `Net` class
    """
        This class simulates behavior of ...
    """
    def __init__(self, 
        x_0: float, # position at the begining of a simualtion  
        y_0: float, 
        power: float, # computing power
        way_equation, # f : (time:float) -> (x:float, y:float)
        direction: int = 1
    ):
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

        self.w = 0
        self.direction = direction

    def __repr__(self):
        return repr(f'{(self.x, self.y)}, {self.power}')

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
        x, y = self.way_equation(self.x, self.y, t)
        self.x = x
        self.y = y

# расстояние между узлами
def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

class Net:
    def __init__(
        self, 
        bandwidth_formula, # bandwidth_formula: max_bandwidth -> (f: distance -> bandwidth)
        nodes: dict
    ):
        self.nodes = nodes  # набор узлов в формате node_id : Node 
        self.G = nx.Graph()  # может иметь произвольное количество компонент связности, должен динамически меняться в зависимотсти от положение узлов
        self.max_bandwidth = 100  # 100 мб/c. Меняет в зависимости от растояния по нелинейным формулам
        self.max_distance = 30  # максимальное расстояние на котором поддерживается свзять 30м. Если расстояние больше, то связь разорвана
        self.bandwidth_formula = bandwidth_formula(self.max_distance) # считаем силу сигнала в зависимости от расстояния по этой формуле. должна учитывать max_bandwidth

        # we need variable to store state of the network.
        # this variable must store information about following:
        # 1. customers and performers of each task
        # 2. sources and destinations of each node

    def move(self, t):
        """
            Call this function to move the nodes according to their 
            way_equation formulas.
        """
        for node in self.nodes:
            node.move(t)

    # update graph's components
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
        # for node in self.nodes:
            # for task ...  # пока есть что скедулить
            #     # вычисляем на ком скедулить
            #     node_dest = scheduler(node, node.tasks[-1])
            #     # вычисляем путь до того, на ком скедулить
            #     rout = net.shortest_path(node, node_dest)
            #     # remove_task(node, -1)
            #     node.rout = rout

    def shortest_path(from_, to_):
        pass # SVYAT


@dataclass(frozen=True)
class Scenario:
    # описание сценария. Передается в конструктор симуляции. Можно написать как вот такой объект, чтобы было наглядно,
    # либо можно написать каждое поле отдельной переменной и передавать в разобранном виде в конструктор
    # data:
    # data:
    # data:
    def __init__(self, coordinates: list[tuple], way_eqs, tasks_list, config_devices: dict, poweroff_posibility: bool):
        self.coordinates = coordinates
        self.way_eqs = way_eqs
        # две возможности задания тасков
        # 1) весь список задач изначально известен. новые задачи не появляются в процессе симуляции
        # 2) задачи генерируются каким-то генератором задач. соотвественно, доп задача напиать генератор задач (от time)
        # т.е появление задач привязано к какому-то времени 
        self.tasks_list = tasks_list # or task generator in 2 case
        self.config_devices = config_devices
        self.poweroff_posibility = poweroff_posibility

class Simulation:
    def __init__(self, steps: int, net: Net, scenario: Scenario, schedule_interval: int):

        # сколько шагов проссимулировать. 1 шаг == 20мс (автоматически дает нам в среднем залержку в 10мс) (180.000 == 1 час)
        self.steps = steps
        self.net = net
        self.schedule_interval = schedule_interval # как часто делаем скедулинг в ms



    # перемещения узлом и многое другое можно визуализировать через анимации. То есть в процессе строить анимацию, а в конце записать ее в файл .gif
    def visualization(self,): 
        # не мы
        pass

    def run(self,):
        for timestep in range(self.steps):
            self.net.update(timestep)
            if timestep % self.schedule_interval == 0:
                self.net.schedule()

def generate_tasks(list_node_ids: list[str])->list[tuple]:
    """
    Для набора id девайсов генерирует список задач, который включает:
    * `calc_size` - вычислительную сложность
    * `transfer_weight` - размер данных для вычислений
    * `transfer_weight_return` - размер ответа
    * `time_to_create` - время появления задачи в симуляции

    На выходе - словарь вида: 
    ```
    {
        id_1: [task_1, task_2, ...],
        id_2: ...,
        ...
    }
    ```
    """
    output = dict()
    for i in list_node_ids:
        count_tasks_node_i = int(np.random.exponential(scale=2.0, size=1)) # количество задач на узел по экспоненциальному распределению
        tasks_node_i = []
        for j in range(count_tasks_node_i):
            calc_size = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение вычислительной сложности задачи?
            transfer_weight = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение размера задачи?
            transfer_weight_return = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение размера ответа
            time_to_create = float(np.random.exponential(scale = 100.0, size=1)) # задаём время, когда должна появится задача по экспоненциальному распределению
            task_j = Task(calc_size, transfer_weight, transfer_weight_return, time_to_create)
            tasks_node_i.append(task_j)
        output[i] = tasks_node_i
    return output


# загружаем json с описанием сценария
number_scenario = 1

file = open(fr'.\scenario\config_scenario_{number_scenario}.json')
scenario = json.load(file)

count_devices = scenario['count_devices']

poweroff_posibility = scenario['poweroff_posibility'] # могут ли девайсы выключиться во время симуляции

nodes = dict()

# Создаём словарь узлов {node_id: Node}
for node_id in scenario['nodes']:
    x0, y0 = scenario['nodes'][node_id]['x'], scenario['nodes'][node_id]['y']
    power = scenario['nodes'][node_id]['power']
    way_eq = scenario['nodes'][node_id]['way_eq']

    # задаём уравнение движения для узла
    if way_eq == "static":
        way_equation = lambda x, y, d, t: (x0, y0)
        # way_equation = lambda t: (x0, y0)
    elif way_eq == "circle":
        w = scenario['nodes'][node_id]['w']
        direction = scenario['nodes'][node_id]['direction'] # её надо хранить в ноде
        xc, yc = scenario['nodes'][node_id]['xc'], scenario['nodes'][node_id]['yc']
        way_equation = lambda x0, y0, d, t: eq_circle(x0, y0, xc, yc, w, d, t)
    elif way_eq == "partline":
        x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
        x_e, y_e =scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
        w, direction = scenario['nodes'][node_id]['w'], 1
        way_equation = lambda x0, y0, d, t: eq_partline(x0, y0, x_s, y_s, x_e, y_e, w, t, d)
    elif way_eq == "sin_or_cos":
        x_s, x_e = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['x_end']
        w = scenario['nodes'][node_id]['w']
        its_sin = scenario['nodes'][node_id]['sin']
        way_equation = lambda x0, y, d, t: eq_sin_or_cos(x0, y0, x_s, x_e, w, t, d, sin=its_sin)
    elif way_eq == "brownian":
        n = 1
        w = scenario['nodes'][node_id]['w']
        x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
        x_e, y_e =scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
        way_equation = lambda x0, y0, d, t: constraints_to_brownian(brownian(x0, y0, n, t, w, out=None), x_s, y_s, x_e, y_e)
    else:
        print(f"Для узла {node_id} задано не реализованное уравнение движения - {way_eq}.")
        raise ValueError

    # задаём узел
    nodes[node_id] = Node(scenario['nodes'][node_id]['x'], scenario['nodes'][node_id]['y'], scenario['nodes'][node_id]['power'], way_equation, direction)

# генерируем сценарии
node_ids = list(scenario['nodes'].keys())
tasks = generate_tasks(node_ids)
# print(tasks)

bandwidth_formula = lambda max_distance: lambda x: 1/x

# задаём сеть
net = Net(bandwidth_formula, nodes)


# 1. Сценарии. Разобрать с генератором задач
# 2. Переменная хранения состояния сети, интерфейс для использования в Simulator (описан в init класса Net)  
# 3. Написать примитивный скедулинг задач. (из описания ниже)
# 4. дописать функцию self.run()
# 5. Визуализация
# 6. Вывод нужной информации в какой-то файлик. Определится с измеряемыми метриками симуляции.
# 7. Подумать над гипотезой о том, что распр. вычисления не эффективны. Подумать как ее проверить (мб александру)

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
