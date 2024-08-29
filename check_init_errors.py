import ast
import os

def check_init_arguments(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())
    
    init_functions = {}
    errors = []
    
    # Walk through all class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for subnode in node.body:
                if isinstance(subnode, ast.FunctionDef) and subnode.name == "__init__":
                    args_count = len(subnode.args.args) - 1  # exclude `self`
                    init_functions[node.name] = args_count

    # Walk through all function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "__init__" and isinstance(node.func.value, ast.Name):
                class_name = node.func.value.id
                if class_name in init_functions:
                    passed_args_count = len(node.args)
                    expected_args_count = init_functions[class_name]
                    if passed_args_count != expected_args_count:
                        errors.append(f"{file_path}: {class_name}.__init__() expects {expected_args_count} arguments but {passed_args_count} were given.")
    
    return errors

def check_all_files(directory):
    error_messages = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                errors = check_init_arguments(file_path)
                if errors:
                    error_messages.extend(errors)
    
    return error_messages

# Define the directory to search
directory_to_search = "E:/AI_Nirvana-1/src"

# Run the check
errors = check_all_files(directory_to_search)
if errors:
    for error in errors:
        print(error)
else:
    print("No errors found related to __init__ argument mismatch.")
