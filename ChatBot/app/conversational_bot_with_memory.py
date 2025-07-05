import os
import json
from typing import TypedDict, Annotated
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, messages_from_dict, messages_to_dict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langsmith import trace
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = 'true'

# Graph State setup
class State(TypedDict):
    messages: Annotated[list, add_messages]
    output: str

# Database Storage Setup
Base = declarative_base()

class MessageStore(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    json_messages = Column(Text)

engine = create_engine("sqlite:///chat_memory.db")
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

class ChatApp():
    def __init__(self, model_name: str, user_id: str, user_role: str, user_text: str):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=300)
        self.chat = self.build_chat_graph()
        self.user_id = user_id
        self.user_role = user_role
        self.user_text = user_text

    def load_user_history(self) -> list[BaseMessage]:
        with SessionLocal() as session:
            record = session.query(MessageStore).filter_by(user_id=self.user_id).first()
            if record:
                return messages_from_dict(json.loads(record.json_messages))
            return []

    def save_user_history(self, messages: list[BaseMessage]):
        with SessionLocal() as session:
            messages_json = json.dumps(messages_to_dict(messages))
            record = session.query(MessageStore).filter_by(user_id=self.user_id).first()
            if record:
                record.json_messages = messages_json
            else:
                record = MessageStore(user_id=self.user_id, json_messages=messages_json)
                session.add(record)
            session.commit()

    def chatbot(self, state: State) -> State:
        reply = self.llm.invoke(state["messages"])
        all_messages = state["messages"] + [reply]
        self.save_user_history(all_messages)
        return {"messages": all_messages, "output": reply.content}

    def build_chat_graph(self):
        builder = StateGraph(State)
        builder.add_node("chatbot", self.chatbot)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)
        return builder.compile(checkpointer=MemorySaver())

    def run(self):
        with trace(project_name="Chatbot", name="Chat Graph",
                   inputs={"role": self.user_role, "text": self.user_text},
                   metadata={"thread_id": self.user_id}) as rt:

            message_history = self.load_user_history()

            thread_id = {"configurable": {"thread_id": self.user_id}}

            message = SystemMessage(self.user_text) if self.user_role.lower() == "system" \
                else HumanMessage(self.user_text)
            message_history.append(message)

            result = self.chat.invoke({"messages": message_history}, thread_id)
            rt.end(outputs={"output": result})
            return result

#if __name__ == "__main__":
    #user_id = "user0004"
    #model = "gpt-4o-mini"

# Instruction
    #instruction = """You are a helpful data analyst who generates SQL queries for users based on their questions.
#Assume you have a table named `sales` with columns `product_id`, `product_name`, and `sales_amount`.
#Generate an SQL query for the user question using the sales table."""

    # Questions that follow the system instruction
    #questions = [
        #"What is the total sales by product?",
        #"What is the total number of products?",
        #"Explain the SQL query you generated."
    #]
    #question_01 = "What is the total sales by product?"
    #question_02 = "What is the total number of products?"
    #question_03 = "Explain the first SQL query you generated."

    # Sets the initial system message
    #app = ChatApp(model_name=model, user_id=user_id, user_role="user", user_text=question_03)
    #response = app.run()
    #print(f"User: {instruction}\nBot: {response['output']}\n{'-'*50}")

    # Sets the initial human message
    #for q in questions:
        #chat_app = ChatApp(model_name=model, user_id=user_id, user_role="user", user_text=q)
        #response = chat_app.run()
        #print(f"User: {q}\nBot: {response['output']}\n{'-'*50}")
    #app = ChatApp(model_name=model, user_id=user_id, user_role="system", user_text=instruction)
    #response = app.run()