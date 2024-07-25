from typing import Callable

import pyttsx3
import speech_recognition as sr


class VoiceInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def listen(self) -> str:
        """监听并识别语音输入"""
        with sr.Microphone() as source:
            print("请说话...")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio, language="zh-CN")
            print(f"您说: {text}")
            return text
        except sr.UnknownValueError:
            print("无法理解音频")
            return ""
        except sr.RequestError as e:
            print(f"无法从Google Speech Recognition服务获取结果; {e}")
            return ""

    def speak(self, text: str) -> None:
        """将文本转换为语音输出"""
        self.engine.say(text)
        self.engine.runAndWait()


def run_voice_interface(self, process_function: Callable[[str], str]) -> None:
    """运行语音交互循环"""
    while True:
        input_text = self.listen()
        if input_text.lower() in ["退出", "结束", "停止"]:
            self.speak("正在退出语音交互。再见！")
            break
        if input_text:
            response = process_function(input_text)
            self.speak(response)
