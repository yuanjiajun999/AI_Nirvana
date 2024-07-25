import unittest
from src.core.task_planner import TaskPlanner, Task


class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner()

    def test_add_task(self):
        self.planner.add_task("Task 1", 1, {"description": "Test task"})
        self.assertEqual(len(self.planner.tasks.queue), 1)

    def test_get_next_task(self):
        self.planner.add_task("Task 1", 2, {"description": "Test task 1"})
        self.planner.add_task("Task 2", 1, {"description": "Test task 2"})
        next_task = self.planner.get_next_task()
        self.assertEqual(next_task.name, "Task 2")

    def test_list_all_tasks(self):
        self.planner.add_task("Task 1", 1, {"description": "Test task 1"})
        self.planner.add_task("Task 2", 2, {"description": "Test task 2"})
        tasks = self.planner.list_all_tasks()
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["name"], "Task 1")
        self.assertEqual(tasks[1]["name"], "Task 2")

    def test_execute_plan(self):
        self.planner.add_task("Task 1", 1, {"description": "Test task 1"})
        self.planner.add_task("Task 2", 2, {"description": "Test task 2"})
        executed_tasks = []

        def execute_function(task):
            executed_tasks.append(task.name)

        self.planner.execute_plan(execute_function)
        self.assertEqual(executed_tasks, ["Task 1", "Task 2"])


if __name__ == "__main__":
    unittest.main()
