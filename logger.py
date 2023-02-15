from structures import Task
class Logger:
    
    def __init__(self, save_path):
        self.save_path = save_path
        
        self.tasks = dict()
    
    def log_task_start(self, time_start, task: Task):
        self.tasks[task] = [time_start,-1]
    
    def log_task_finish(self, time_end, task: Task):
        if task in self.started:
            self.tasks[task][1] = time_end
    
    def print_logs(self,):
        for task in self.tasks.keys():
            time_start, time_end = self.tasks[task]
            print(f'TASK: {task}\n\tTIME START: {time_start}\n\tTIME FINISHED: {time_end}')

