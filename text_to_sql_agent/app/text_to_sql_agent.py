from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langsmith import trace
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'

# --- Structured Output Model Class Definition ---
class SQLModel(BaseModel):
    """The model will create SQL executable statement for the user query."""
    sql_query: str = Field(..., description="An executable SQL statement to answer user question")


# --- State Class Definition ---
class QueryState(TypedDict):
    user_query: str
    sql_query: str
    query_explanation: str


# --- Main Class ---
class TextToSQLAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.sql_llm = self.llm.with_structured_output(SQLModel)
        self.explain_llm = ChatOpenAI(model=model_name, temperature=0.7, max_tokens=500)

        self.generate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful data analyst who generates SQL queries for users based on their questions.
Assume you have a table named `sales` with columns `product_id`, `product_name`, and `sales_amount`.
Generate an SQL query for the user question using the sales table."""),
            ("human", "Question: {question}")
        ])

        self.explain_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant who explains SQL queries to users."),
            ("human", "sql_query: {sql_query}")
        ])

        self.graph = self._build_graph()

    def _generate_sql(self, state: QueryState) -> QueryState:
        from operator import itemgetter
        chain = itemgetter("question") | self.generate_prompt | self.sql_llm
        sql_result = chain.invoke({"question": state["user_query"]})
        print("Generated SQL:", sql_result.sql_query)
        return {
            "user_query": state["user_query"],
            "sql_query": sql_result.sql_query,
            "query_explanation": ""
        }

    def _generate_explanation(self, state: QueryState) -> QueryState:
        from operator import itemgetter
        chain = itemgetter("sql_query") | self.explain_prompt | self.explain_llm
        explain_result = chain.invoke({"sql_query": state["sql_query"]})
        return {
            "user_query": state["user_query"],
            "sql_query": state["sql_query"],
            "query_explanation": explain_result.content
        }

    def _build_graph(self):
        builder = StateGraph(QueryState)
        builder.add_node("generate_sql", self._generate_sql)
        builder.add_node("generate_explanation", self._generate_explanation)
        builder.add_edge(START, "generate_sql")
        builder.add_edge("generate_sql", "generate_explanation")
        builder.add_edge("generate_explanation", END)
        return builder.compile()

    def run(self, question: str) -> QueryState:
        with trace(project_name="Text to SQL", name="Generate and Explain SQL",
                   inputs={"question": question}) as rt:
            result = self.graph.invoke({"user_query": question})
            rt.end(outputs={"output": result})
        return result
