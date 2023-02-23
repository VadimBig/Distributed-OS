from dataclasses import dataclass
import networkx as nx
import numpy as np
from collections import deque
from typing import Callable, Iterable
from trajectory_equations import eq_circle, eq_partline, eq_sin_or_cos, brownian, constraints_to_brownian
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm

@dataclass(frozen=True)  # (можно просто tuple или как удобнее)
class Task:
    calc_size: float  # размер задачи для выполнения
    transfer_weight: int  # размер задачи для передачи данных в байтах
    transfer_weight_return: int  # размер результата задачи для передачи данных
    customer_id: int  # id of a customer of the task
    task_id: int
    # time_to_create: float # время появления задачи

    def __hash__(self) -> int:
        return self.transfer_weight + self.transfer_weight_return + self.customer_id


class Logger:
    def __init__(self, save_path):
        self.save_path = save_path

        self.tasks = dict()

    def log_task_start(self, time_start, task: Task):
        self.tasks[task] = [time_start, -1]

    def log_task_finish(self, time_end, task: Task):
        if task in self.tasks:
            self.tasks[task][1] = time_end

    def print_logs(self,):
        for task in self.tasks.keys():
            time_start, time_end = self.tasks[task]
            print(
                f'TASK: {task}\n\tTIME GIVEN TO THE NET: {time_start}\n\tTIME FINISHED: {time_end}')
    
    def stats(self,):
        # calc mean, std, 
        # calc_durs = [self.tasks[task][1] - self.tasks[task][0] for task in self.tasks.keys()]
        # calc_durs = [x for x in calc_durs if x > 0]
        df = pd.DataFrame()
        ids = []
        calc_size = []
        transfer_weight = []
        return_weight = []
        customer_id = []
        time_given = []
        time_finished = []
        for task in self.tasks.keys():
            ids.append(task.task_id)
            calc_size.append(task.calc_size)
            transfer_weight.append(task.transfer_weight)
            return_weight.append(task.transfer_weight_return)
            customer_id.append(task.customer_id)
            time_given.append(self.tasks[task][0])
            time_finished.append(self.tasks[task][1])
        df['id'] = ids
        df['calc_size'] = calc_size
        df['transfer_weight'] = transfer_weight
        df['transfer_return_weight'] = return_weight
        df['customer_id'] = customer_id
        df['time_given'] = time_given
        df['time_finished'] = time_finished
        return df      


class Node:
    # SUGGESTION FOR ALEXANDER:
    # I think that sources and destinations of each node must be stored in the `Net` class
    """
        This class simulates behavior of ...
    """

    def __init__(self,
                 x0,
                 y0,
                 power: float,  # computing power
                 # f : (time:float) -> (x:float, y:float)
                 way_equation: Callable,
                 ):
        self.direction = 1
        self.x, self.y = x0, y0
        self.power = power
        self.isActive = True  # True if the node is capable of interacting with others
        self.isCalculating = False  # True if the node is computing some task
        self.isTransfering = False  # True if the node is transfering some data
        self.way_equation = way_equation
        self.tasks = deque()  # queue of tasks for node to compute
        self.given_tasks = deque()  # queue for tasks which are given to another customer
        self.current_progress = 0

    def __update_state(self, ):
        self.isCalculating = bool(self.tasks)

    def forget_tasks(self, to_forget: Iterable):
        new_tasks = deque()
        forgotten_tasks = []
        for task in self.tasks:
            if task.customer_id not in to_forget:
                new_tasks.append(task)
            else:
                forgotten_tasks.append(task)
        self.tasks = new_tasks
        return forgotten_tasks

    def del_task(self, task: Task, from_given=False):
        if task in self.tasks:
            self.tasks.remove(task)
        if from_given and task in self.given_tasks:
            self.given_tasks.remove(task)
        self.__update_state()

    def give_task(self, task: Task):
        self.del_task(task)
        self.__update_state()

    def remember_given_task(self, task):
        self.add_task(task)
        self.__update_state()

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

    def get_est_start_executions(self,):
        '''
            Returns list of tasks and its estimated start of execution (ESOE)
            If task is in process of computation, then ESOE equals its negative current progress.
        '''
        eet = []
        loading = self.get_loading()
        tasks_rev = self.tasks.copy()
        tasks_rev.reverse()
        for task in tasks_rev:
            loading -= task.calc_size
            eet.append((task, loading))
        return eet

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
        self.x, self.y, self.direction = self.way_equation(
            self.x, self.y, self.direction, t)

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
        logger: Logger,
        debug_info=True,
        mode='basic'
    ):
        self.debug_info = debug_info
        self.nodes = nodes  # набор узлов в формате node_id : Node
        self.G = nx.Graph()  # может иметь произвольное количество компонент связности, должен динамически меняться в зависимотсти от положение узлов
        # 100 мб/c. Меняет в зависимости от растояния по нелинейным формулам
        self.max_bandwidth = 37.5
        # максимальное расстояние на котором поддерживается свзять 13м. Если расстояние больше, то связь разорвана
        self.max_distance = 30
        # считаем силу сигнала в зависимости от расстояния по этой формуле. должна учитывать max_bandwidth
        self.bandwidth_formula = bandwidth_formula(
            self.max_distance, self.max_bandwidth)
        self.to_be_sent_back = []
        self.transfers = []
        self.logger = logger
        self.mode = mode

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
                key_i, key_j = keys[i], keys[j]
                x, y = self.nodes[key_i], self.nodes[key_j]
                d = dist(x, y)
                self.G.add_edge(key_i, key_j, weight=self.bandwidth_formula(d))
                if d >= self.max_distance:
                    self.G.remove_edge(key_i, key_j)

    def __check_connection(self, id_1, id_2):

        if nx.has_path(self.G, id_1, id_2):
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
        # call this function to begin transfer of task, or results of its computation
        end_time = timestep + \
            self.__calc_transfer_time(task, bandwidth, is_result=is_result)
        self.transfers.append((task, route, end_time, is_result))
        if len(route) > 1:
            for node_id in route:
                self.nodes[node_id].start_transfer()

    def __stop_transfering(self,
                           transfer,
                           finished,
                           timestep,
                           is_result=False
                           ):
        # stop transfering,
        # (if transfering was successfull)
        # If a task was tranfered, we call `add_task` to assign this task to a new performer
        # If a computation results were transfered, we call `SOME_FUNCTION` to finish task lifetime
        task, route, end_time, is_result = transfer
        if len(route) > 1:
            for node_id in route:
                self.nodes[node_id].end_transfer()
        if finished:
            if is_result:
                self.__finish_task(route[0], task,timestep=timestep)
            else:
                self.add_task(route[-1], task)

    def add_task(self, performer_id, task: Task):
        # assign task to perfortmer
        self.nodes[performer_id].add_task(task)
        if self.debug_info:
            print(f"TASK `{task}` WAS GIVEN TO THE NODE {performer_id}")

    def __finish_task(self, performer_id, task, timestep):
        # call this function in case your task is done and results were transfered back to customer
        customer_id = task.customer_id
        self.nodes[performer_id].del_task(task)
        self.nodes[customer_id].del_task(task, from_given=True)
        # LOGS
        self.logger.log_task_finish(timestep, task)

    def update(self, timestep, timedelta):
        # update general state of net
        self.move(timedelta)
        self.update_components()
        # transferings update
        if self.debug_info:
            print("TO BE SENT BACK: ", self.to_be_sent_back)
        transfers_to_continue = []
        transfers_to_stop = []
        transfers_calc_results_to_finish = []
        transfers_to_finish = []

        for transfer in self.transfers:
            task, route, end_time, is_result = transfer
            # CHECK IF THE TRANSFERS ARE STILL HAPPENING
            if end_time <= timestep:
                if is_result:
                    transfers_calc_results_to_finish.append(transfer)
                else:
                    transfers_to_finish.append(transfer)
            # CHECK IF THE TRANSFERS ARE STILL POSSIBLE
            break_loop = False
            for i in range(len(route)):
                for j in range(i+1, len(route)):
                    if not self.__check_connection(route[i], route[j]):
                        transfers_to_stop.append(transfer)
                        # exit from outer loop
                        break_loop = True
                        break
                if break_loop:
                    break
        transfers_to_continue = [
            transfer for transfer in self.transfers if transfer not in transfers_to_stop and transfer not in transfers_to_finish and transfer not in transfers_calc_results_to_finish]

        # debug
        if self.debug_info:
            print('TRANSFERS TO CONTINUE:', transfers_to_continue)
            print('TRANSFERS TO FINISH:', transfers_to_finish)
            print('TRANSFERS TO STOP', transfers_to_stop)

        self.transfers = transfers_to_continue
        for transfer in transfers_to_stop:
            self.__stop_transfering(transfer, finished=False,timestep=timestep)
        for transfer in transfers_to_finish:
            self.__stop_transfering(transfer, finished=True,timestep=timestep)
        for transfer in transfers_calc_results_to_finish:
            self.__stop_transfering(transfer, finished=True, is_result=True,timestep=timestep)

        # START TRANSFER RESULTS OF COMPUTATIONS
        not_sended = []
        for performer_id, finished_task in self.to_be_sent_back:
            customer_id = finished_task.customer_id
            if customer_id != performer_id:
                route, cost = self.shortest_path(performer_id, customer_id)
                # print(route, cost)
                if cost == -1:  # if there are no path between performer and customer?
                    not_sended.append((performer_id, finished_task))
                else:
                    self.__start_transfering(
                        finished_task, cost, route, timestep, is_result=True)
            else:
                self.__finish_task(performer_id, finished_task, timestep)
        self.to_be_sent_back = not_sended

        for node_id in self.nodes.keys():

            # забываем о тех задачах, которые были выданы узлами, покинувшими компоненту связности узла
            lost_connections = set()
            for task in self.nodes[node_id].tasks:
                customer_id = task.customer_id
                if not self.__check_connection(node_id, customer_id):
                    lost_connections.add(customer_id)
            if lost_connections:
                forgotten_tasks = self.nodes[node_id].forget_tasks(
                    lost_connections)
                for ftask in forgotten_tasks:
                    self.nodes[customer_id].remember_given_task(ftask)
            # collect data to be sent back
            finished_task = self.nodes[node_id].calc(timedelta)
            if finished_task:
                self.to_be_sent_back.append((node_id, finished_task))

        # return 0
        # 1) проверить что передача данных все еще идет и все в порядке (нет разрыва сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
        # 2) проверить что вычисления все еще имеют смысл (тот для кого мы вычисляем все еще в сети). Иначе - обработать ситуацию (РЕАЛИЗОВАНО)
        # 3) если первые два условия не выполняются, то можно написать алгоритмы опитмизации,
        #    чтобы по возращению устрйоств в сеть они продоолжали выполнять прерванное (НЕ РЕАЛИЗОВАНО)
    def schedule_all(self,
                     timestep,
                     ):
        if self.mode == 'basic':
            scheduler = self.basic_scheduler
        elif self.mode == 'elementary':
            scheduler = self.elementaty_scheduler
        else:
            print(
                "Wrong value encountered in `mode` argument. Possible options are: basic, elementary")
            raise ValueError

        if self.debug_info:
            print('------------ SCHEDULING STARTED ------------')
        for node_id in self.nodes.keys():
            eses = self.nodes[node_id].get_est_start_executions()
            for task, est_start_time in eses:
                min_cost = task.calc_size / \
                    self.nodes[node_id].power + est_start_time
                opt_performer, route, _, route_bandwidth = scheduler(
                    node_id, task, min_cost=min_cost)
                if opt_performer != node_id:
                    self.nodes[node_id].give_task(task)
                    self.__start_transfering(
                        task, route_bandwidth, route, timestep)
                    if self.debug_info:
                        print('TASK TRANSFERING STARTED:', task, route)
        if self.debug_info:
            print('------------ SCHEDULING FINISHED ------------')

    def schedule(self,
                 timestep,  # current time (in the simulation) # TIMESTEP
                 to_schedule,  # list of tuples (node_id, Task)
                 ):

        # A PROBLEM TO DISCUSS:
        # приоритет в постановке задач получают те ноды, которые идут первыми в списке этой итерации
        if self.mode == 'basic':
            scheduler = self.basic_scheduler
        elif self.mode == 'elementary':
            scheduler = self.elementaty_scheduler
        else:
            print(
                "Wrong value encountered in `mode` argument. Possible options are: basic, elementary")
            raise ValueError

        if self.debug_info:
            print('------------ SCHEDULING STARTED ------------')
        for node_id, task in to_schedule:
            self.logger.log_task_start(task=task, time_start=timestep)
            opt_performer, route, _, route_bandwidth = scheduler(
                node_id=node_id, task=task)
            self.__start_transfering(task, route_bandwidth, route, timestep)
            if self.debug_info:
                print('TASK TRANSFERING STARTED:', task, route)
        if self.debug_info:
            print('------------ SCHEDULING FINISHED ------------')

    def basic_scheduler(self,
                        node_id,
                        task: Task,
                        min_cost=None):
        '''
            Scheduler which assigns tasks to nodes according to the following algorithm:
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
        if min_cost is None:
            min_cost = self.nodes[node_id].get_loading(
            ) + task.calc_size / self.nodes[node_id].power
        optimal_performer = node_id
        route_to_performer = [node_id]
        route_bandwidth = float('inf')
        for p_id in all_reachable_nodes:
            route, route_cost = self.shortest_path(node_id, p_id)
            if route_cost < 0:
                continue
            overall_cost = (task.transfer_weight + task.transfer_weight_return)/route_cost + \
                self.nodes[p_id].get_loading() + task.calc_size / \
                self.nodes[p_id].power
            if self.debug_info:
                print('OVERALL_COST: ', overall_cost, 'NODE ID: ', p_id)
            if overall_cost < min_cost:
                min_cost = overall_cost
                optimal_performer = p_id
                route_to_performer = route
                route_bandwidth = route_cost
        if self.debug_info:
            print(
                f"OPTIMAL PERFORMER {optimal_performer}, OPTIMAL ROUTE {route_to_performer}")
        return optimal_performer, route_to_performer, min_cost, route_bandwidth

    def elementaty_scheduler(self,
                             node_id,
                             task: Task,
                             min_cost=None
                             ):
        '''
            A simplest scheduler which assigns every node its task
            input:
                'node_id' (int) - id of a customer node
                'task' (Task) - a task to be scheduled
            output:
                'node_id' (int) - an id of the task performer
        '''
        if min_cost is None:
            min_cost = self.nodes[node_id].get_loading(
            ) + task.calc_size / self.nodes[node_id].power
        return node_id, [node_id], min_cost, float('inf')

    def shortest_path(self, from_, to_):
        self.update_components()
        D = nx.DiGraph()
        for i in self.G.nodes:
            for j in self.G.nodes:
                if (i, j) in self.G.edges and not (self.nodes[i].isTransfering or self.nodes[j].isTransfering):
                    D.add_edge(i, j, weight=self.G.edges[i, j]['weight'])
                    D.add_edge(j, i, weight=self.G.edges[i, j]['weight'])

        for i in nx.weakly_connected_components(D):
            if (from_ in i) and (to_ in i):
                cost = dict.fromkeys(D.nodes, -1)
                cost[from_] = float("Inf")
                vertexes = dict.fromkeys(D.nodes)
                for i in range(D.number_of_nodes() - 1):
                    for u, v, w in D.edges(data=True):
                        if cost[u] != -1 and min(w['weight'], cost[u]) > cost[v]:
                            cost[v] = min(w['weight'], cost[u])
                            vertexes[v] = u
                route = [to_]
                r = to_
                while r != from_:
                    r = vertexes[r]
                    route.insert(0, r)
                # route.append(cost[to_])
                return route, cost[to_]
        return [from_], -1


class Simulation:
    def __init__(self, step: float, net: Net, tasks, logger: Logger):
        # TASKS - LIST OF TUPLES: (CUSTOMER_ID, TIME_TO_CREATE, TASK)
        # сколько шагов проссимулировать. 1 шаг == 20мс (автоматически дает нам в среднем залержку в 10мс) (180.000 == 1 час)
        self.step = step  # шаг симуляции
        self.net = net
        self.logger = logger
        self.tasks = deque(tasks)
        self.ALL_TASKS = deque(tasks)
        self.time = 0
        # self.schedule_interval = schedule_interval  # как часто делаем скедулинг в ms

    # перемещения узлом и многое другое можно визуализировать через анимации. То есть в процессе строить анимацию, а в конце записать ее в файл .gif

    def visualization(self, save_dir, boundaries):
        x_down, y_down,x_up,y_up = boundaries
        plt.xlim([x_down, x_up]) 
        plt.ylim([y_down, y_up]) 
        task_nodes=[]
        routes_edges=[]
        for task in self.net.transfers:
            route=task[1]
            customer=route[0]
            performer=route[-1]
            task_nodes.extend([customer,performer])
            for i in range(len(route)-1):
                routes_edges.append([route[i],route[i+1]])
        
        for u,v in self.net.G.edges():
            Xu,Yu=self.net.nodes[u].x,self.net.nodes[u].y
            Xv,Yv=self.net.nodes[v].x,self.net.nodes[v].y
            if [u,v] in routes_edges :
                plt.plot([Xu,Xv],[Yu,Yv],'g',linewidth=2)
                plt.arrow(Xu,Yu,(Xv-Xu)/2,(Yv-Yu)/2,length_includes_head=True,head_length = 0.5,head_width = 0.3,color='g')
            elif [v,u] in routes_edges:
                plt.plot([Xu,Xv],[Yu,Yv],'g',linewidth=2)
                plt.arrow(Xv,Yv,(Xu-Xv)/2,(Yu-Yv)/2,length_includes_head=True,head_length = 0.5,head_width = 0.3,color='g')
            else:
                plt.plot([Xu,Xv],[Yu,Yv],'b',linewidth=1)
                    

        for node_id in self.net.nodes.keys():
            x,y = self.net.nodes[node_id].x, self.net.nodes[node_id].y
            if node_id in task_nodes:
                plt.plot(x,y,'ro',markersize=12)
            elif self.net.nodes[node_id].isCalculating:
                plt.plot(x,y,'rp',markersize=12)
            else:
                plt.plot(x,y,'bo',markersize=12)
            plt.text(x,y,f"{node_id}",horizontalalignment='center', 
               verticalalignment='center',fontweight='bold',size=7,color='black')
        plt.title(f'time={self.time}')
        cur_dt = datetime.now()
        plt.savefig(f'{save_dir}/image_{cur_dt.month}-{cur_dt.day}-{cur_dt.hour}-{cur_dt.minute}-{cur_dt.second}-{cur_dt.microsecond}.png',dpi=200)
        plt.clf()

    def run(self, sim_time,boundaries, save_dir=None, describe_file=None):
        '''
            Start simualtion.
            sim_time -- time of a simulation in ms
        '''
        if save_dir:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        steps = sim_time // self.step
        with tqdm(total=steps) as pbar:
            while self.time <= sim_time:
                self.time += self.step
                self.net.update(self.time, self.step)
                curr_tasks = self.__current_tasks()
                self.net.schedule(timestep=self.time, to_schedule=curr_tasks)
                if self.time % 10000 == 0:
                    self.net.schedule_all(timestep=self.time)
                if save_dir and self.time % 300 == 0:
                    self.visualization(boundaries=boundaries,save_dir=save_dir)
                pbar.update(1)
        self.logger.print_logs()
        if save_dir:
            self.__make_gif(save_dir)
        if describe_file:
            self.logger.stats().to_csv(describe_file)

    def __make_gif(self,frames_dir):
        # получим список с именами всех картинок, находящихся в папке
        files = os.listdir(frames_dir)
        files = [os.path.join(frames_dir, f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        #quantity_of_Frames = len(frames_Name)
        frames = [Image.open(img_path) for img_path in files if img_path[-3:]=='png']
        frames[0].save(
            frames_dir+'simulation.gif',
            save_all=True,
            append_images=frames[1:],  # Срез который игнорирует первый кадр.
            optimize=True,
            duration=120,
            loop=0
        )
        for f in files:
            if f[-3:]=='png':
                os.remove(f)

    def reset(self,):
        '''
            Resets simulation
        '''
        self.time = 0
        # self.net.restart()
        self.tasks = self.ALL_TASKS

    def __current_tasks(self,):
        current_tasks = []
        while self.tasks and self.tasks[0][1] <= self.time:
            current_tasks.append((self.tasks[0][0], self.tasks[0][2]))
            self.tasks.popleft()
        return current_tasks
