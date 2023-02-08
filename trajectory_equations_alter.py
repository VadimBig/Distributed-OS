def eq_circle(x0, y0, xc, yc, w, direction):
    """
    Уравнение движения по окружности
    * `x0`, `y0` - координаты точки в текущий момент
    * `xc`, `yc` - координаты центра окружности
    * `w` - угловая скорость
    * `t` - момент времени
    * `direction` - направление движение: либо 1, либо -1

    На выход - `lambda t: (x1(t), y1(t))`
    """

    # phi0 = math.atan((yc - y0) / (xc - x0))
    r = ((x0 - xc)**2 + (y0 - yc)**2)**0.5
    phi0 = math.acos((yc - y0) / r)

    return lambda t: (r * math.cos(phi0 + w * t * direction), r * math.sin(phi0 + w * t * direction))

def eq_partline(x_start, y_start, x_end, y_end, v):
    """
    Уравнение движения вдоль отрезка
    * `x_start`, `y_start` - координаты начала отрезка
    * `x_end`, `y_end` - координаты конца отрезка
    * `x0`, `y0` - текущее положение точки
    * v - скорость точки
    * t - время

    На выход - `lambda t: (x1(t), y1(t))`
    """
    phi = math.atan((y_start - y_end) / (x_start - x_end))

    x_len, y_len = abs(x_end - x_start), abs(y_end - y_start)

    f = lambda t: (x_start + (math.cos(phi) * v * t) % x_len, y_start + (math.sin(phi) * v * t) % y_len) \
        if ((math.cos(phi) * v * t) // x_len) % 2 == 0 \
        else (x_end - (math.cos(phi) * v * t) % x_len, y_end - (math.sin(phi) * v * t) % y_len)
    
    return f

def eq_sin_or_cos(x_start, x_end, y0, v, sin=True):
    """
    Уравнение движения вдоль синуса/косинуса (неравномерное)
    * `x_start`, `x_end` - координаты начала и конца синуса/косинуса
    * `y0` - значение, на которое поднимаем синус/косинус по оси абсцисс
    * `v` - скорость движения по оси x
    * `t` - время
    * `sin` - булева переменная. Если `True`, то `y1=sin(x1)`, иначе `y1=cos(x1)`

    На выход - `lambda t: (x1(t), y1(t))`
    """
    
    x_len = abs(x_end - x_start)
    
    if sin == True:
        f = lambda t: (x_start + (v * t) % x_len, math.sin(x_start + (v * t) % x_len) + y0) \
            if ((v * t) // x_len) % 2 == 0 \
            else (x_end - (v * t) % x_len, math.sin(x_end - (v * t) % x_len) + y0)
    else:
        f = lambda t: (x_start + (v * t) % x_len, math.cos(x_start + (v * t) % x_len) + y0) \
            if ((v * t) // x_len) % 2 == 0 \
            else (x_end - (v * t) % x_len, math.cos(x_end - (v * t) % x_len) + y0)
    
    return f
