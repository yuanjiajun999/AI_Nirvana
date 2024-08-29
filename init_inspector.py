import inspect
import traceback

def check_init_args(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        sig = inspect.signature(original_init)
        expected_args = len(sig.parameters) - 1  # Exclude 'self'
        provided_args = len(args) + len(kwargs)

        if provided_args != expected_args:
            print(f"Error initializing {cls.__name__}: Expected {expected_args} arguments but {provided_args} were given.")
            print("Call stack:")
            for line in traceback.format_stack():
                print(line.strip())

        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls

def patch_module(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            setattr(module, name, check_init_args(obj))

def main():
    # 导入你项目的主要模块
    import src.main as main_module
    
    # 在模块中打补丁
    patch_module(main_module)
    
    # 提供必要的参数
    config_file = "path_to_your_config_file"  # 替换为你的配置文件路径
    mode = "cli"  # 确保这里传递的是实际需要的模式

    # 启动应用程序
    try:
        main_module.main(config_file, mode)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
