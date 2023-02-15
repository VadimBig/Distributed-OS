from main import Node, Net, Task

# no moving equations
def get_constant_wayeq(x0, y0): return lambda t: (x0, y0)
const_way_eq1 = get_constant_wayeq(0,0)
const_way_eq2 = get_constant_wayeq(5,0)
const_way_eq3 = get_constant_wayeq(-5,0)  
const_way_eq4 = get_constant_wayeq(10,0)  
const_way_eq5 = get_constant_wayeq(25,0)  

# test nodes
node1 = Node(power=10,way_equation=const_way_eq1)
node2 = Node(power=1000,way_equation=const_way_eq2)
node3 = Node(power=100,way_equation=const_way_eq3)
node4 = Node(power=100,way_equation=const_way_eq4)
node5 = Node(power=100,way_equation=const_way_eq5)
nodes_dict = {
    1: node1,
    2: node2,
    3: node3,
    4: node4,
    5: node5
}

# test bandwidth formula
bandwidth_formula = lambda max_dist, max_bandwidth: (lambda d: max_bandwidth - d*(max_bandwidth/max_dist)) 

# test Net
net = Net(bandwidth_formula=bandwidth_formula, nodes=nodes_dict)

# test tasks
task1 = Task(calc_size=1000000, transfer_weight=1000,transfer_weight_return=1,customer_id=1)
task2 = Task(calc_size=1000, transfer_weight=1000,transfer_weight_return=1000,customer_id=2)
task3 = Task(calc_size=1000, transfer_weight=1000,transfer_weight_return=1000,customer_id=3)
task4 = Task(calc_size=1000, transfer_weight=1000,transfer_weight_return=1000,customer_id=4)

# check if updates work as they should
net.update_components()
# print(net.G.edges)

# check add_task method
# net.add_task(1,task=task1)
# print(net.nodes[1])

# check scheduling (1 task must go to 2nd node)
net.schedule(0,to_schedule=[(1,task1)])
net.update(3,3)
net.update(12,12-3)

"""
TODO
1) Добавить следующую функцианальность. Если узел, считающий нашу задачу покинул сеть, задача должна быть "возвращена" в стек заказчика
2) Изменять пропускную способность узла в зависимости от количества передач между ним?
"""

