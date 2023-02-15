import copy
from structures import *

def get_brown(x_s, y_s, x_e, y_e, w, n=1):
    def way_equation(x0, y0, d, t): return constraints_to_brownian(
                brownian(x0, y0, n, t, w), x_s, y_s, x_e, y_e)

    return way_equation

def get_partline(x_s, y_s, x_e, y_e, w):
    def way_equation(x0, y0, d, t): return eq_partline(
                x0, y0, x_s, y_s, x_e, y_e, w, t, d)

    return way_equation

def get_static(x0, y0, d=1):
    def way_equation(x0, y0, d, t): return (x0, y0, d)

    return way_equation

def get_circle(xc, yc, w, d=1):
    def way_equation(x0, y0, d, t): return eq_circle(
                x0, y0, xc, yc, w, d, t)
    
    return way_equation

def get_sincos(x_s, x_e, y, w, sin):
    def way_equation(x0, y0, d, t): return eq_sin_or_cos(
                x0, y0, x_s, x_e, y, w, t, d, sin)
    
    return way_equation

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
        # количество задач на узел по экспоненциальному распределению
        count_tasks_node_i = int(np.random.exponential(scale=2.0, size=1))
        for j in range(count_tasks_node_i):
            # распределение вычислительной сложности задачи?
            calc_size = int(
                np.abs(np.random.normal(loc=150.0, scale=100, size=1)))
            # распределение размера задачи?
            transfer_weight = int(
                np.abs(np.random.normal(loc=150.0, scale=100, size=1)))
            transfer_weight_return = int(np.abs(np.random.normal(
                loc=150.0, scale=100, size=1)))  # распределение размера ответа
            # задаём время, когда должна появится задача по экспоненциальному распределению
            time_to_create = prev_time + \
                float(np.random.exponential(scale=100.0, size=1))
            task_j = Task(calc_size, transfer_weight,
                          transfer_weight_return, int(i))
            output.append((i, time_to_create, task_j))
            prev_time = time_to_create
    return sorted(output, key=lambda x: x[1])


if __name__ == "__main__":
    # задаём мощность классов узлов
    maxpower = 1
    powers = {
        1: maxpower,
        2: maxpower * 0.65,
        3: maxpower * 0.35,
        4: maxpower * 0.25,
        5: maxpower * 0.2,
        6: maxpower * 0.125,
        7: maxpower * 0.075,
        8: maxpower * 0.01
    }
    # загружаем json с описанием сценария
    number_scenario = '2'
    np.random.seed(0)
    file = open(fr'.\scenario\config_scenario_{number_scenario}.json')
    scenario = json.load(file)

    count_devices = scenario['count_devices']

    # могут ли девайсы выключиться во время симуляции
    poweroff_posibility = scenario['poweroff_posibility']

    nodes = dict()
    way_equations = [0] * (len(scenario['nodes']) + 1)
    # Создаём словарь узлов {node_id: Node}
    for node_id in scenario['nodes']:

        x0, y0 = scenario['nodes'][node_id]['x'], scenario['nodes'][node_id]['y']
        power = powers[scenario['nodes'][node_id]['class_power']]
        way_eq = scenario['nodes'][node_id]['way_eq']

        # задаём уравнение движения для узла
        if way_eq == "static":
            # def way_equation(x0, y0, d, t): return (x0, y0, d)
            # way_equation = lambda x0, y0, d, t: (x0, y0, d)
            # way_equation = lambda x0, y0, d, t: (x0, y0, d)
            wq = get_static(x0, y0)
        elif way_eq == "circle":
            w = scenario['nodes'][node_id]['w']
            # её надо хранить в ноде
            direction = scenario['nodes'][node_id]['direction']
            xc, yc = scenario['nodes'][node_id]['xc'], scenario['nodes'][node_id]['yc']

            # def way_equation(x0, y0, d, t): return eq_circle(
            #     x0, y0, xc, yc, w, d, t)

            wq = get_circle(xc, yc, w, direction)
            # way_equation = lambda x0, y0, d, t: eq_circle(
            #     x0, y0, xc, yc, w, d, t)
        elif way_eq == "partline":
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
            w, direction = scenario['nodes'][node_id]['w'], scenario['nodes'][node_id]['direction']

            # def way_equation(x0, y0, d, t): return eq_partline(
            #     x0, y0, x_s, y_s, x_e, y_e, w, t, d)

            wq = get_partline(x_s, y_s, x_e, y_e, w)
            # way_equation = lambda x0, y0, d, t: eq_partline(
            #     x0, y0, copy.deepcopy(x_s), copy.deepcopy(y_s), copy.deepcopy(x_e), copy.deepcopy(y_e), copy.deepcopy(w), t, d)

        elif way_eq == "sin_or_cos":
            x_s, x_e = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['x_end']
            w = scenario['nodes'][node_id]['w']
            its_sin = scenario['nodes'][node_id]['sin']
            y = y0

            # def way_equation(x0, y0, d, t): return eq_sin_or_cos(
            #     x0, y0, x_s, x_e, y, w, t, d, sin=its_sin)

            wq = get_sincos(x_s, x_e, y, w, its_sin)
            # way_equation = lambda x0, y0, d, t: eq_sin_or_cos(
            #     x0, y0, x_s, x_e, y, w, t, d, sin=its_sin)
        elif way_eq == "brownian":
            n = 1
            w = scenario['nodes'][node_id]['w']
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']

            # def way_equation(x0, y0, d, t): return constraints_to_brownian(
            #     brownian(x0, y0, n, t, w), x_s, y_s, x_e, y_e)

            wq = get_brown(x_s, y_s, x_e, y_e, w)
            # way_equation = lambda x0, y0, d, t: constraints_to_brownian(
            #     brownian(x0, y0, n, t, w), x_s, y_s, x_e, y_e)
            # way_equations[int(node_id)] = lambda x0, y0, d, t: constraints_to_brownian(
            #     brownian(x0, y0, n, t, w), scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start'], scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end'])
        else:
            print(
                f"Для узла {node_id} задано не реализованное уравнение движения - {way_eq}.")
            raise ValueError
        # # задаём узел
        # nodes[int(node_id)] = Node(scenario['nodes'][node_id]['x'], scenario['nodes']
        #                            [node_id]['y'], scenario['nodes'][node_id]['power'], way_equation)
        nodes[int(node_id)] = Node(x0, y0, power, wq)

    print(nodes)
    # генерируем сценарии
    node_ids = [int(a) for a in list(scenario['nodes'].keys())]
    tasks = generate_tasks(node_ids)
    # print(tasks)

    def bandwidth_formula(max_dist, max_bandwidth): return (
        lambda d: max_bandwidth - d*(max_bandwidth/max_dist))
    logger = Logger('text.txt')
    # задаём сеть
    net = Net(bandwidth_formula, nodes,logger=logger,debug_info=False,mode='basic')
    # customer, time, task = tasks[0]
    # net.update(0,0)
    # print(task)
    # print(net.G.edges)
    # net.schedule(time,to_schedule=[(task.customer_id, task)])
    # net.update(103.9,103.9)
    # net.update(104,0.1)
    # print(net.nodes)
    sim = Simulation(tasks=tasks, net=net, step=10,logger=logger)
    sim.run(20000,(-10, -10, 10, 10),save_dir="test_1") # 200

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
