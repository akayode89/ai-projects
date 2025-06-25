from fastapi import FastAPI
from pydantic import BaseModel
from app.text_to_sql_agent import TextToSQLAgent

app = FastAPI()
agent = TextToSQLAgent()

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def generate_sql(query: QueryRequest):
    result = agent.run(query.question)
    return result
