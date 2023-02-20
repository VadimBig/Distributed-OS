from structures import *
import json


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

def generate_tasks(list_node_ids: list[str], expect_tasks_on_one=2.0, std=0.5 ** 0.5) -> list[tuple]:
    """
    Для набора id девайсов генерирует список задач, который включает:
    * `calc_size` - вычислительную сложность
    * `transfer_weight` - размер данных для вычислений
    * `transfer_weight_return` - размер ответа
    * `time_to_create` - время появления задачи в симуляции
    * `expect_tasks_on_one` - матожидание задач на один узел
    * `std` - стандартное отклонение, если используется нормально распределение, а не экспоненциальное для генерации задач

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
        # count_tasks_node_i = int(np.random.exponential(scale=expect_tasks_on_one, size=1))
        
        # количество задач на узел по нормальному распределению
        count_tasks_node_i = int(np.abs(np.random.normal(loc=expect_tasks_on_one, scale=std, size=1)))
        
        classes_tasks = np.random.choice(4, count_tasks_node_i, p=[0.3, 0.4, 0.2, 0.1])
        for j in range(count_tasks_node_i):
            class_task_j = classes_tasks[j]
            if class_task_j == 0:
                # вычислительная сложность
                beta = 0.5
                max_calc_size = 1200000

                # отправка
                max_send_size = 200
                sigma_send_size = 10

                # получение
                max_get_size = 200
                sigma_get_size = 10

            elif class_task_j == 1:
                # вычислительная сложность
                beta = 0.5
                max_calc_size = 600000

                # отправка
                max_send_size = 100
                sigma_send_size = 7

                # получение
                max_get_size = 100
                sigma_get_size = 7

            elif class_task_j == 2:
                # вычислительная сложность
                beta = 0.5
                max_calc_size = 240000

                # отправка
                max_send_size = 30
                sigma_send_size = 7

                # получение
                max_get_size = 30
                sigma_get_size = 7

            else:
                # вычислительная сложность
                beta = 0.5
                max_calc_size = 5000

                # отправка
                max_send_size = 10.0
                sigma_send_size = 5

                # получение
                max_get_size = 10.0
                sigma_get_size = 5

            # распределение вычислительной сложности задачи
            calc_size = int((np.random.exponential(beta, size=1) % 1.0) * max_calc_size)

            # распределение размера задачи
            transfer_weight = int(
                np.abs(np.random.normal(loc=max_send_size, scale=sigma_send_size, size=1)))
            
            # распределение размера ответа
            transfer_weight_return = int(np.abs(np.random.normal(
                loc=max_get_size, scale=sigma_get_size, size=1)))

            # задаём время, когда должна появится задача по экспоненциальному распределению
            time_to_create = prev_time + \
                float(np.random.exponential(scale=1000.0, size=1))
            task_j = Task(calc_size, transfer_weight,
                          transfer_weight_return, int(i))
            output.append((i, time_to_create, task_j))
            prev_time = time_to_create
    return sorted(output, key=lambda x: x[1])


if __name__ == "__main__":
    # задаём мощность классов узлов
    maxpower = 1
    powers = {
        1: maxpower, # ноутбук мощный
        2: maxpower * 0.65, # ноутбук средний
        3: maxpower * 0.35, # смартфон мощный
        4: maxpower * 0.25, # планшет средний
        5: maxpower * 0.2, # смартфон средний
        6: maxpower * 0.125, # смартфон бюджетный
        7: maxpower * 0.075, # смарт-телевизор
        8: maxpower * 0.01 # смарт-часы
    }
    # загружаем json с описанием сценария
    number_scenario = '1'
    mode = 'basic'
    sim_time = 10_000_000 # ms
    vis=False
    np.random.seed(1)
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
            wq = get_static(x0, y0)

        elif way_eq == "circle":
            w = scenario['nodes'][node_id]['w']
            direction = scenario['nodes'][node_id]['direction']
            xc, yc = scenario['nodes'][node_id]['xc'], scenario['nodes'][node_id]['yc']
            wq = get_circle(xc, yc, w, direction)
        elif way_eq == "partline":
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
            w, direction = scenario['nodes'][node_id]['w'], scenario['nodes'][node_id]['direction']
            wq = get_partline(x_s, y_s, x_e, y_e, w)

        elif way_eq == "sin_or_cos":
            x_s, x_e = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['x_end']
            w = scenario['nodes'][node_id]['w']
            its_sin = scenario['nodes'][node_id]['sin']
            y = y0
            wq = get_sincos(x_s, x_e, y, w, its_sin)

        elif way_eq == "brownian":
            n = 1
            w = scenario['nodes'][node_id]['w']
            x_s, y_s = scenario['nodes'][node_id]['x_start'], scenario['nodes'][node_id]['y_start']
            x_e, y_e = scenario['nodes'][node_id]['x_end'], scenario['nodes'][node_id]['y_end']
            wq = get_brown(x_s, y_s, x_e, y_e, w)

        else:
            print(
                f"Для узла {node_id} задано не реализованное уравнение движения - {way_eq}.")
            raise ValueError

        nodes[int(node_id)] = Node(x0, y0, power, wq)

    # генерируем сценарии
    node_ids = [int(a) for a in list(scenario['nodes'].keys())]
    tasks = generate_tasks(node_ids)

    def bandwidth_formula(max_dist, max_bandwidth): return (
        lambda d: max_bandwidth - d*(max_bandwidth/max_dist))
    logger = Logger('text.txt')
    # задаём сеть
    net = Net(bandwidth_formula, nodes,logger=logger,debug_info=False,mode=mode)
    # customer, time, task = tasks[0]
    # net.update(0,0)
    # print(task)
    # print(net.G.edges)
    # net.schedule(time,to_schedule=[(task.customer_id, task)])
    # net.update(103.9,103.9)
    # net.update(104,0.1)
    # print(net.nodes)
    # format X_min, Y_min, X_max, Y_max
    boundaries = {
        '1': (-5, -9, 9, 8),
        '2': (-3*np.sqrt(2), -3*np.sqrt(2), 3*np.sqrt(2), 3*np.sqrt(2)),
        '3_1': (-4, -6, 4, 6),
        '3_2': (-12, -12, 12, 12),
        '4':(-4, -4, 4, 4)
    }


    sim = Simulation(tasks=tasks, net=net, step=10,logger=logger)
    if vis==True:
        save_dir = f'sim_results/test_{number_scenario}_{mode}'
    else:
        save_dir=None
    sim.run(sim_time,boundaries=boundaries[number_scenario],save_dir=None,describe_file=f'sim_results/test_{number_scenario}_{mode}.csv') # 200

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
