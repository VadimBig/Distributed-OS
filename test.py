class Node: 
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def __repr__(self,):
        return str(self.x) + ' ' + str(self.y)
##test

a = Node(5, [1])
b = Node(10, [2])

d = {1: a.y, 2: b.y}

d[1].append(2)

print(a)
b.x = 14

print(d)