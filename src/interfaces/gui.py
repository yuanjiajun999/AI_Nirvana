import sys

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)


class AINirvanaGUI(QMainWindow):
    def __init__(self, ai_nirvana):
        super().__init__()
        self.ai_nirvana = ai_nirvana
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('AI Nirvana')
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.input_text = QTextEdit()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.on_generate)

        layout.addWidget(self.input_text)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.output_text)

    def on_generate(self):
        input_text = self.input_text.toPlainText()
        result = self.ai_nirvana.process(input_text)
        self.output_text.setPlainText(str(result))

def run_gui(ai_nirvana):
    app = QApplication(sys.argv)
    window = AINirvanaGUI(ai_nirvana)
    window.show()
    sys.exit(app.exec_())