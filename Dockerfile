FROM ai_nirvana_base:latest

   WORKDIR /app

   COPY src/ ./src/
   COPY requirements.txt .

   CMD ["python", "src/main.py"]