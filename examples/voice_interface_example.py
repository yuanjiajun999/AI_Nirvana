from src.interfaces.voice_interface import VoiceInterface


def process_command(command):
    # 这里应该是实际的命令处理逻辑
    return f"正在处理命令：{command}"


def main():
    voice_interface = VoiceInterface()

    print("开始语音交互，说'退出'来结束程序。")
    voice_interface.run_voice_interface(process_command)


if __name__ == "__main__":
    main()
