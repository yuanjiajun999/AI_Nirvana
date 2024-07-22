import cProfile
import pstats
import io
from src.main import main
from src.config import config  # 假设配置在 config.py 中

def profile_main():
    pr = cProfile.Profile()
    pr.enable()
    
    # 运行主程序，传入 config
    main(config)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

if __name__ == "__main__":
    profile_main()