from src.interfaces.gui import AINirvanaGUI, run_gui
from src.main import AINirvana
from src.config import Config


def main():
    config = Config()
    ai_nirvana = AINirvana(config)
    run_gui(ai_nirvana)


if __name__ == "__main__":
    main()
