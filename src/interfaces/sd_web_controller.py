import time
import tkinter as tk
from tkinter import simpledialog

from ai_nirvana.utils.config import Config
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class AINorvanaController:
    def __init__(self):
        self.config = Config()
        sd_url = self.config.get("sd_web_ui", {}).get("url", "http://localhost:7860")
        self.driver = webdriver.Chrome()  # 或其他浏览器驱动
        self.driver.get(sd_url)
        self.wait = WebDriverWait(self.driver, 10)

        self.root = tk.Tk()
        self.root.title("AI Norvana Controller")
        self.root.geometry("300x100")

        self.start_button = tk.Button(
            self.root, text="Start AI Control", command=self.start_ai_control
        )
        self.start_button.pack()

        self.interrupt_button = tk.Button(
            self.root,
            text="Interrupt",
            command=self.interrupt_ai_control,
            state=tk.DISABLED,
        )
        self.interrupt_button.pack()

        self.is_ai_controlling = False

    def start_ai_control(self):
        user_input = simpledialog.askstring("Input", "What would you like to generate?")
        if user_input:
            self.is_ai_controlling = True
            self.start_button.config(state=tk.DISABLED)
            self.interrupt_button.config(state=tk.NORMAL)
            self.generate_image(user_input)

    def interrupt_ai_control(self):
        self.is_ai_controlling = False
        self.start_button.config(state=tk.NORMAL)
        self.interrupt_button.config(state=tk.DISABLED)

    def generate_image(self, prompt):
        try:
            # 定位并清除提示输入框
            prompt_box = self.wait.until(
                EC.presence_of_element_located((By.ID, "txt2img_prompt"))
            )
            prompt_box.clear()

            # 输入提示
            prompt_box.send_keys(prompt)

            # 点击生成按钮
            generate_button = self.wait.until(
                EC.element_to_be_clickable((By.ID, "txt2img_generate"))
            )
            generate_button.click()

            # 等待图像生成完成
            while self.is_ai_controlling:
                progress = self.driver.find_element(By.ID, "txt2img_progressbar")
                if "display: none;" in progress.get_attribute("style"):
                    break
                time.sleep(0.5)

            if self.is_ai_controlling:
                print("Image generation completed!")
            else:
                print("Image generation interrupted by user.")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            self.interrupt_ai_control()

    def run(self):
        self.root.mainloop()


def run_sd_web_controller():
    controller = AINorvanaController()
    controller.run()


if __name__ == "__main__":
    run_sd_web_controller()
