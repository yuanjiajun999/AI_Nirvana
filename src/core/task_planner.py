from queue import PriorityQueue
from typing import Any, Dict, List


class Task:
    def __init__(self, name: str, priority: int, details: Dict[str, Any]):
        self.name = name
        self.priority = priority
        self.details = details

    def __lt__(self, other):
        return self.priority < other.priority


class TaskPlanner:
    def __init__(self):
        self.tasks = PriorityQueue()


def add_task(self, name: str, priority: int, details: Dict[str, Any]) -> None:
    """添加任务到规划器"""
    task = Task(name, priority, details)
    self.tasks.put(task)


def get_next_task(self) -> Task:
    """获取下一个优先级最高的任务"""
    if not self.tasks.empty():
        return self.tasks.get()
    else:
        raise ValueError("没有更多的任务")


def list_all_tasks(self) -> List[Dict[str, Any]]:
    """列出所有任务"""
    all_tasks = []
    temp_queue = PriorityQueue()
    while not self.tasks.empty():
        task = self.tasks.get()
        all_tasks.append(
            {"name": task.name, "priority": task.priority, "details": task.details}
        )
        temp_queue.put(task)
    self.tasks = temp_queue
    return all_tasks


def execute_plan(self, execute_function):
    """执行任务计划"""
    while not self.tasks.empty():
        task = self.get_next_task()
        execute_function(task)
