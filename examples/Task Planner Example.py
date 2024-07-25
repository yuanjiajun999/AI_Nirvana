from src.core.task_planner import TaskPlanner


def execute_task(task):
    print(f"Executing task: {task.name}")
    print(f"Task details: {task.details}")
    print(f"Task priority: {task.priority}")
    print("Task executed successfully")
    print()


def main():
    planner = TaskPlanner()

    # 添加任务
    planner.add_task(
        "Data Analysis", 2, {"dataset": "sales_data.csv", "method": "regression"}
    )
    planner.add_task("Model Training", 1, {"model": "neural_network", "epochs": 100})
    planner.add_task(
        "Report Generation",
        3,
        {"format": "pdf", "sections": ["summary", "results", "conclusion"]},
    )

    # 列出所有任务
    all_tasks = planner.list_all_tasks()
    print("All tasks:")
    for task in all_tasks:
        print(f"Name: {task['name']}, Priority: {task['priority']}")
    print()

    # 执行任务计划
    print("Executing task plan:")
    planner.execute_plan(execute_task)


if __name__ == "__main__":
    main()
