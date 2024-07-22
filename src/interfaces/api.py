from ai_nirvana.main import AINirvana
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
ai_nirvana = AINirvana()

class Query(BaseModel):
    text: str

@app.post("/process")
async def process_query(query: Query):
    try:
        response = ai_nirvana.process(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)