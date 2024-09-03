import time
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return f"Hello! The current time is: {time.strftime('%H:%M:%S')}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')