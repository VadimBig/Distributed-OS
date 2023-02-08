from dataclasses import dataclass
import networkx as nx
import numpy as np
from collections import deque
from typing import Callable, Iterable
from brownian import brownian, constraints_to_brownian
from trajectory_equations import eq_circle, eq_partline, eq_sin_or_cos
import json
import math


@dataclass(frozen=True)  # (можно просто tuple или как удобнее)
class Task:
    calc_size: float  # размер задачи для выполнения
    transfer_weight: int  # размер задачи для передачи данных в байтах
    transfer_weight_return: int  # размер результата задачи для передачи данных
    customer_id: int  # id of a customer of the task
    # time_to_create: float # время появления задачи


class Node:
    # SUGGESTION FOR ALEXANDER:
    # I think that sources and destinations of each node must be stored in the `Net` class
    """
        This class simulates behavior of ...
    """

    def __init__(self,  
                 power: float,  # computing power
                 # f : (time:float) -> (x:float, y:float)
                 way_equation: Callable,
                 ):
        self.x, self.y = way_equation(0)
        self.power = power
        self.isActive = True  # True if the node is capable of interacting with others
        self.isCalculating = False  # True if the node is computing some task
        self.isTransfering = False  # True if the node is transfering some data
        self.way_equation = way_equation
        self.tasks = deque()  # queue of tasks for node to compute
        self.given_tasks = deque() # queue for tasks which are given to another customer
        self.current_progress = 0

        # self.w = 0
        # self.direction = direction

    def __update_state(self, ):
        self.isCalculating = bool(self.tasks)

    def forget_tasks(self, to_forget: Iterable):
        print('forgetting tasks: ', to_forget)
        new_tasks = deque()
        for task in self.tasks:
            if task.customer_id not in to_forget:
                new_tasks.append(task)
        self.tasks = new_tasks

    def calc(self, timedelta: float) -> Task:
        """
            This method simulates process of node computing the tasks. 
            timedelta: float - computing time
        """
        self.current_progress += self.power * timedelta
        finished_task = None
        if self.tasks and self.current_progress - self.tasks[0].calc_size >= 0:
            self.current_progress -= self.tasks[0].calc_size
            finished_task = self.tasks.popleft()
        elif not self.tasks:
            self.current_progress = 0
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

    def get_loading(self,):
        '''
            Returns the current loading of the node.

            Loading is (a sum of all task sizes in a queue of tasks minus current progress) divided by a computing power of the node 
        '''
        return sum([t.calc_size for t in self.tasks]) - self.current_progress

    def wake(self, ):
        assert self.isActive == False, "This node isn't sleeping."
        self.isActive = True

    def sleep(self, ):
        assert self.isActive == True, "This node is already sleeping."
        self.isActive = False

    def start_transfer(self, ):
        assert self.isTransfering == False, "This node is already transfering some data."
        self.isTransfering = True

    def end_transfer(self, ):
        assert self.isTransfering == True, "This node isn't transfering any data now."
        self.isTransfering = False

    def move(self, t):
        x, y = self.way_equation(t)
        self.x = x
        self.y = y

    def __str__(self) -> str:
        d = {'x': self.x,
             'y': self.y,
             'power': self.power,
             'isActive': self.isActive,
             'isCalculating': self.isCalculating,
             'isTransfering': self.isTransfering,
             'tasks': str(self.tasks),
             'current_progress': self.current_progress
             }
        return str(d)
    def __repr__(self):
        return self.__str__()

# расстояние между узлами


def dist(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


class Net:
    def __init__(
        self,
        # bandwidth_formula: max_distance, max_bandwidth -> (f: distance -> bandwidth)
        bandwidth_formula,
        nodes: dict,
        debug_info = True
    ):
        self.debug_info = True
        self.nodes = nodes  # набор узлов в формате node_id : Node
        self.G = nx.Graph()  # может иметь произвольное количество компонент связности, должен динамически меняться в зависимотсти от положение узлов
        # 100 мб/c. Меняет в зависимости от растояния по нелинейным формулам
        self.max_bandwidth = 100
        # максимальное расстояние на котором поддерживается свзять 30м. Если расстояние больше, то связь разорвана
        self.max_distance = 30
        # считаем силу сигнала в зависимости от расстояния по этой формуле. должна учитывать max_bandwidth
        self.bandwidth_formula = bandwidth_formula(self.max_distance, self.max_bandwidth)

        # we need variable to store state of the network.
        # this variable must store information about following:
        # 1. customers and performers of each task
        # 2. sources and destinations of each node

        self.task_customers = {}  # {task_id : node_id (node is a customer)}
        # {node_id: list of ids of performers}
        self.performers = Net.__init_performers(self.nodes.keys())
        # {node_id: destination of data transmission} # if destination is 0, then a node doesn't transimit data
        self.destinations = Net.__init_destinations(self.nodes.keys())
        self.to_be_sent_back = []
        self.transfers = []

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
        for node_id in self.nodes.keys():
            self.nodes[node_id].move(t)

    # update graph's components
    def update_components(self,):
        # при переопределении ребра, информация о прошлом ребре стирается
        keys = list(self.nodes.keys())
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                key_i = keys[i]
                key_j = keys[j]
                x = self.nodes[key_i]
                y = self.nodes[key_j]
                d = dist(x, y)
                self.G.add_edge(key_i, key_j, weight=self.bandwidth_formula(d))
                if d >= self.max_distance:
                    self.G.remove_edge(key_i, key_j)

    def __check_connection(self, id_1, id_2):
        if nx.shortest_path(self.G, id_1, id_2):
            return True
        return False

    def __calc_transfer_time(self,
                               task: Task,
                               bandwidth,
                               is_result=False
                               ):
        if is_result:
            return task.transfer_weight_return/bandwidth
        return task.transfer_weight/bandwidth

    def __start_transfering(self,
                            task: Task,
                            bandwidth,
                            route,
                            timestep,
                            is_result=False
                            ):
        end_time = timestep + self.__calc_transfer_time(task, bandwidth, is_result=is_result)
        self.transfers.append((task, route, end_time, is_result))
        for node_id in route:
            self.nodes[node_id].start_transfer()

    def __stop_transfering(self,
                           transfer,
                           finished
                           ):
        task, route, end_time, is_result = transfer
        for node_id in route:
            self.nodes[node_id].end_transfer()
        if finished: 
            self.add_task(route[-1], task)

    def add_task(self, customer_id, task: Task):
        self.nodes[customer_id].add_task(task)
        if self.debug_info: print(f"TASK `{task}` WAS GIVEN TO THE NODE {customer_id}")
    
    def __finish_task(self, task):
        customer_id = task.customer_id
        self.nodes[customer_id].forget_tasks([task])

    def update(self, timestep, timedelta):

        self.move(timestep)
        self.update_components()
        # transferings update
        transfers_to_continue = []
        transfers_to_stop = []
        transfers_calc_results_to_finish = []
        transfers_to_finish = []
        for transfer in self.transfers:
            task, route, end_time, is_result = transfer
            # CHECK IF THE TRANSFERS ARE STILL HAPPENING
            if end_time <= timestep:
                if task is None:
                    transfers_calc_results_to_finish.append(transfer)
                else:
                    transfers_to_finish.append(transfer)
            # CHECK IF THE TRANSFERS ARE STILL POSSIBLE
            for i in range(len(route)):
                for j in range(i+1, len(route)):
                    if not self.__check_connection(route[i], route[j]):
                        transfers_to_stop.append(transfer)
                        break
        transfers_to_continue = [transfer for transfer in self.transfers if transfer not in transfers_to_stop and transfer not in transfers_to_finish]
        
        ## debug
        if self.debug_info:
            print('TRANSFERS TO CONTINUE:', transfers_to_continue)
            print('TRANSFERS TO FINISH:', transfers_to_finish)
            print('TRANSFERS TO STOP', transfers_to_stop)

        self.transfers = transfers_to_continue
        for transfer in transfers_to_stop:
            self.__stop_transfering(transfer,finished=False)
        for transfer in transfers_to_finish:
            self.__stop_transfering(transfer,finished=True)

        # TRANSFER RESULTS OF COMPUTATIONS
        not_sended = []
        for performer_id, finished_task in self.to_be_sent_back:
            customer_id = finished_task.customer_id
            if customer_id != performer_id:
                route, cost = self.shortest_path(performer_id, customer_id)
                if cost == -1:
                    not_sended.append((performer_id, finished_task))
                else:
                    self.__start_transfering(finished_task, cost, route, timestep, is_results=True)
            else:
                self.__finish_task(finished_task)
        self.to_be_sent_back = not_sended

        for node_id in self.nodes.keys():
            if self.destinations[node_id]:
                is_connected = self.check_connection(
                    node_id, self.destinations[node_id])
                if not is_connected:
                    self.destinations[node_id] = 0

            # забываем о тех задачах, которые были выданы узлами, покинувшими компоненту связности узла
            lost_connections = set()    
            for task in self.nodes[node_id].tasks:
                customer_id = task.customer_id
                if not self.__check_connection(node_id, customer_id):
                    lost_connections.add(customer_id)
            if lost_connections: self.nodes[node_id].forget_tasks(lost_connections)
            # collect data to be sent back
            finished_task = self.nodes[node_id].calc(timedelta)
            if finished_task:
                self.to_be_sent_back.append((node_id, finished_task))

        # return 0
        # 1) проверить что передача данных все еще идет и все в порядке (нет разрыва сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
        # 2) проверить что вычисления все еще имеют смысл (тот для кого мы вычисляем все еще в сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
        # 3) если первые два условия не выполняются, то можно написать алгоритмы опитмизации,
        #    чтобы по возращению устрйоств в сеть они продоолжали выполнять прерванное (НЕ РЕАЛИЗОВАНО)

    def schedule(self,
                 timestamp,  # current time (in the simulation)
                 to_schedule,  # list of tuples (node_id, Task)
                 mode='basic',  # possible values: basic, elementary
                 ):

        # A PROBLEM TO DISCUSS:
        # приоритет в постановке задач получают те ноды, которые идут первыми в списке этой итерации
        if mode == 'basic':
            scheduler = self.basic_scheduler
        elif mode == 'elementary':
            scheduler = self.elementaty_scheduler
        else:
            print(
                "Wrong value encountered in `mode` argument. Possible options are: basic, elementary")
            raise ValueError

        if self.debug_info: print('------------ SCHEDULING STARTED ------------')
        for node_id, task in to_schedule:
            opt_performer, route, _, route_bandwidth = scheduler(
                node_id=node_id, task=task)
            self.__start_transfering(task, route_bandwidth, route, timestamp)
            if self.debug_info: print('TASK TRANSFERING STARTED:', task, route)
        if self.debug_info: print('------------ SCHEDULING FINISHED ------------')
        


    def basic_scheduler(self,
                        node_id,
                        task: Task):
        '''
            A basic scheduler which assigns tasks to nodes according to the following algorithm


            в компоненте связности каждому ребру присваиваем вес transfer_weight / bandwidth
            запускаем беллмана форда для source -> получаем расстояния до всех других вершин
            добавляем к каждму расстоянию calc_size / power
            выбираем наименьшее число из полученных - это тот кто выполнит нашу задачу быстрее всех с учетом передачи данных

            так как нам еще нужно вернуть результат в source, то на первом этапе прибавляем еще аналогичное слагаемое для transfer_weight_return


            input:
                'node_id' (int) - id of a customer node
                'task' (Task) - a task to be scheduled
            output:
                'node_id' (int) - an id of the task performer
        '''
        all_reachable_nodes = list(nx.shortest_path(self.G, node_id).keys())
        min_cost = self.nodes[node_id].get_loading() + task.calc_size / self.nodes[node_id].power
        optimal_performer = node_id
        route_to_performer = [node_id]
        route_bandwidth = float('inf')
        for p_id in all_reachable_nodes:
            route, route_cost = self.shortest_path(node_id, p_id)
            if route_cost < 0:
                continue
            overall_cost = (task.transfer_weight + task.transfer_weight_return)/route_cost + \
                self.nodes[p_id].get_loading() + task.calc_size / self.nodes[p_id].power
            if self.debug_info: print('OVERALL_COST: ', overall_cost, 'NODE ID: ', p_id)
            if overall_cost < min_cost:
                min_cost = overall_cost
                optimal_performer = p_id
                route_to_performer = route
                route_bandwidth = route_cost
        if self.debug_info: print(f"OPTIMAL PERFORMER {optimal_performer}, OPTIMAL ROUTE {route_to_performer}")
        return optimal_performer, route_to_performer, min_cost, route_bandwidth

    def elementaty_scheduler(self,
                             node_id,
                             task: Task
                             ):
        '''
            A simplest scheduler which assigns every node its task
            input:
                'node_id' (int) - id of a customer node
                'task' (Task) - a task to be scheduled
            output:
                'node_id' (int) - an id of the task performer
        '''
        return node_id, [node_id], self.nodes[node_id].get_loading(), float('inf')

    def shortest_path(self,from_, to_):
        self.update_components()
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
                route=[[to_]]
                r=to_
                while r != from_:
                    r=vertexes[r]
                    route[0].insert(0,r)
                route.append(cost[to_])
                return route
            
        return [[from_], -1]


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
        self.tasks_list = tasks_list  # or task generator in 2 case
        self.config_devices = config_devices
        self.poweroff_posibility = poweroff_posibility


class Simulation:
    def __init__(self, steps: int, net: Net, scenario: Scenario, schedule_interval: int):

        # сколько шагов проссимулировать. 1 шаг == 20мс (автоматически дает нам в среднем залержку в 10мс) (180.000 == 1 час)
        self.steps = steps
        self.net = net
        self.schedule_interval = schedule_interval  # как часто делаем скедулинг в ms

    # перемещения узлом и многое другое можно визуализировать через анимации. То есть в процессе строить анимацию, а в конце записать ее в файл .gif

    def visualization(self,):
        # не мы
        pass
    

    def create_nodes(self, scenario_nodes):
        """
        Принимает на вход `nodes` - словарь, содержащий информацию об узлах, вида:
        ```
        "nodes": {
            "1": {
                "x": -2,
                "y": -2,
                "x_start": -4,
                "y_start": -4,
                "x_end": 4,
                "y_end": 4,
                "power": 10,
                "w": 10,
                "way_eq": "brownian",
                "direction": 1
        },
        ```
        На выходе словарь вида:
        ```
        {
            node_id: Node
        }
        ```
        """
        nodes = dict()
        # Создаём словарь узлов {node_id: Node}
        for node_id in scenario_nodes:
            x0, y0 = scenario_nodes[node_id]['x'], scenario_nodes[node_id]['y']
            power = scenario_nodes[node_id]['power']
            way_eq = scenario_nodes[node_id]['way_eq']

            # задаём уравнение движения для узла
            if way_eq == "static":
                way_equation = lambda x, y, d, t: (x0, y0)
                # way_equation = lambda t: (x0, y0)
            elif way_eq == "circle":
                w = scenario_nodes[node_id]['w']
                direction = scenario_nodes[node_id]['direction'] # её надо хранить в ноде
                xc, yc = scenario_nodes[node_id]['xc'], scenario_nodes[node_id]['yc']
                way_equation = lambda x0, y0, d, t: eq_circle(x0, y0, xc, yc, w, d, t)
            elif way_eq == "partline":
                x_s, y_s = scenario_nodes[node_id]['x_start'], scenario_nodes[node_id]['y_start']
                x_e, y_e = scenario_nodes[node_id]['x_end'], scenario_nodes[node_id]['y_end']
                w, direction = scenario_nodes[node_id]['w'], 1
                way_equation = lambda x0, y0, d, t: eq_partline(x0, y0, x_s, y_s, x_e, y_e, w, t, d)
            elif way_eq == "sin_or_cos":
                x_s, x_e = scenario_nodes[node_id]['x_start'], scenario_nodes[node_id]['x_end']
                w = scenario_nodes[node_id]['w']
                its_sin = scenario_nodes[node_id]['sin']
                way_equation = lambda x0, y, d, t: eq_sin_or_cos(x0, y0, x_s, x_e, w, t, d, sin=its_sin)
            elif way_eq == "brownian":
                n = 1
                w = scenario_nodes[node_id]['w']
                x_s, y_s = scenario_nodes[node_id]['x_start'], scenario_nodes[node_id]['y_start']
                x_e, y_e = scenario_nodes[node_id]['x_end'], scenario_nodes[node_id]['y_end']
                way_equation = lambda x0, y0, d, t: constraints_to_brownian(brownian(x0, y0, n, t, w, out=None), x_s, y_s, x_e, y_e)
            else:
                print(f"Для узла {node_id} задано не реализованное уравнение движения - {way_eq}.")
                raise ValueError

            # задаём узел
            nodes[node_id] = Node(x0, y0, power, way_equation, direction)

    def run(self,):
        for timestep in range(self.steps):
            self.net.update(timestep)
            if timestep % self.schedule_interval == 0:
                self.net.schedule()


def generate_tasks(list_node_ids: list[str]) -> list[tuple]:
    """
    Для набора id девайсов генерирует список задач, который включает:
    * `calc_size` - вычислительную сложность
    * `transfer_weight` - размер данных для вычислений
    * `transfer_weight_return` - размер ответа
    * `time_to_create` - время появления задачи в симуляции

    На выходе отсортированный по времени список: 
    ```
    [
        ('1', , time_to_create_1, task_1),
        ('1', , time_to_create_2, task_2),
        ...
    ]
    ```
    """
    output = []
    prev_time = 0.0
    for i in list_node_ids:
        count_tasks_node_i = int(np.random.exponential(scale=2.0, size=1)) # количество задач на узел по экспоненциальному распределению
        for j in range(count_tasks_node_i):
            calc_size = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение вычислительной сложности задачи?
            transfer_weight = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение размера задачи?
            transfer_weight_return = int(np.abs(np.random.normal(loc=150.0, scale=100, size=1))) # распределение размера ответа
            time_to_create = prev_time + float(np.random.exponential(scale = 100.0, size=1)) # задаём время, когда должна появится задача по экспоненциальному распределению
            task_j = Task(calc_size, transfer_weight, transfer_weight_return)
            output.append((i, time_to_create, task_j))
            prev_time = time_to_create
    return sorted(output, key=lambda x: x[1])


if __name__ == "__main__":
    # загружаем json с описанием сценария
    number_scenario = 1

    file = open(fr'.\scenario\config_scenario_{number_scenario}.json')
    scenario = json.load(file)

    count_devices = scenario['count_devices']

    # могут ли девайсы выключиться во время симуляции
    poweroff_posibility = scenario['poweroff_posibility']

    nodes = dict()

    # Создаём словарь узлов {node_id: Node}
    for node_id in scenario['nodes']:
        x0, y0 = scenario['nodes'][node_id]['x'], scenario['nodes'][node_id]['y']
        power = scenario['nodes'][node_id]['power']
        way_eq = scenario['nodes'][node_id]['way_eq']

        # задаём уравнение движения для узла
        if way_eq == "static":
            def way_equation(x, y, d, t): return (x0, y0)
            # way_equation = lambda t: (x0, y0)
        elif way_eq == "circle":
            w = scenario['nodes'][node_id]['w']
            # её надо хранить в ноде
            direction = scenario['nodes'][node_id]['direction']
            xc, yc = scenario['nodes'][node_id]['xc'], scenario['nodes'][node_id]['yc']

            def way_equation(x0, y0, d, t): return eq_circle(
                x0, y0, xc, yc, w, d, t)
        elif way_eq == "partline":
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
            w, direction = scenario['nodes'][node_id]['w'], 1

            def way_equation(x0, y0, d, t): return eq_partline(
                x0, y0, x_s, y_s, x_e, y_e, w, t, d)
        elif way_eq == "sin_or_cos":
            x_s, x_e = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['x_end']
            w = scenario['nodes'][node_id]['w']
            its_sin = scenario['nodes'][node_id]['sin']

            def way_equation(x0, y, d, t): return eq_sin_or_cos(
                x0, y0, x_s, x_e, w, t, d, sin=its_sin)
        elif way_eq == "brownian":
            n = 1
            w = scenario['nodes'][node_id]['w']
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
            def way_equation(x0, y0, d, t): return constraints_to_brownian(
                brownian(x0, y0, n, t, w, out=None), x_s, y_s, x_e, y_e)
        else:
            print(
                f"Для узла {node_id} задано не реализованное уравнение движения - {way_eq}.")
            raise ValueError

        # задаём узел
        nodes[node_id] = Node(scenario['nodes'][node_id]['x'], scenario['nodes'][node_id]
                            ['y'], scenario['nodes'][node_id]['power'], way_equation, direction)

    # генерируем сценарии
    node_ids = list(scenario['nodes'].keys())
    tasks = generate_tasks(node_ids)
    # print(tasks)


    def bandwidth_formula(max_distance): return lambda x: 1/x


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
