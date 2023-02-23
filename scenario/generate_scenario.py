import json

def generateScenario(k: int, name:str="config_scenario"):
    """
    Генерирует json-file с 13-ю точками в комнате [-0.5*`k`, 0.5*`k`] x [-0.5*`k`, 0.5*`k`]

    * `k` - коэффициент масштаба
    """

    data = {
        "count_devices": 13,
        "nodes": {
            "1": {
                "x": 0.0 * k,
                "y": 0.25 * k,
                "class_power": 1,
                "way_eq": "static"
            },
            "2": {
                "x": -0.45 * k,
                "y": -0.45 * k,
                "class_power": 2,
                "way_eq": "static"
            },
            "3": {
                "x": 0.25 * k,
                "y": -0.45 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 3,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "4": {
                "x": -0.4 * k,
                "y": 0.45 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 3,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "5": {
                "x": -0.4 * k,
                "y": 0.15 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 4,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "6": {
                "x": 0.2 * k,
                "y": 0.4 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 4,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "7": {
                "x": -0.45 * k,
                "y": -0.2 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 5,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "8": {
                "x": 0.45 * k,
                "y": 0.3 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 5,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "9": {
                "x": -0.45 * k,
                "y": 0.4 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 6,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "10": {
                "x": -0.3 * k,
                "y": 0.4 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 6,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "11": {
                "x": -0.25 * k,
                "y": 0.2 * k,
                "class_power": 7,
                "way_eq": "static"
            },
            "12": {
                "x": 0.05 * k,
                "y": -0.45 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 8,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            },
            "13": {
                "x": 0.45 * k,
                "y": 0.4 * k,
                "x_start": -0.5 * k,
                "y_start": -0.5 * k,
                "x_end": 0.5 * k,
                "y_end": 0.5 * k,
                "class_power": 8,
                "w": 0.0008,
                "way_eq": "brownian",
                "direction": 1
            }

        },
        "boundaries": [(-0.5 * k, 0.5 * k), (-0.5 * k, 0.5 * k)],
        "poweroff_posibility": False

    }

    # сохраняем json сценария
    with open(f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


generateScenario(42.4, "config_scenario_2")

generateScenario(42.4, "config_scenario_3")

generateScenario(42.4, "config_scenario_4")

