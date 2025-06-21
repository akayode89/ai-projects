from pydantic import BaseModel, Field
from typing import Literal
from langsmith import traceable, trace
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class RetrieverApp():
    def __init__(self):
        self.embedding_model_name = "text-embedding-3-small"
        self.embeddings = "text-embedding-3-small"
        self.collection_name = "routes_collection"

    def doc_load(self, path):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        return loader.load()

    def doc_split(self, docs):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langsmith import trace
        with trace(name="doc_split", input={"doc_count": len(docs), "doc": docs }) as ls:
            splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=['\n\n\n', '\n\n'])
            document_split = splits.split_documents(docs)
            ls.end(outputs={"split_count": len(document_split)})
            return document_split

    def doc_store(self, db_location, docs):
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        print(1)
        embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name=self.collection_name,
            persist_directory=db_location
        )
        print(2)
        vector_store.reset_collection()
        vector_store.add_documents(documents=docs)
        return vector_store

    def GetRetriever(self, file_path, db_location):
        doc = self.doc_load(file_path)
        document_split = self.doc_split(doc)
        vector_store = self.doc_store(db_location=db_location, docs=document_split)
        return vector_store.as_retriever(search_kwargs={'k': 2})

class RouteModel(BaseModel):
    """Route user request to the relevant data source"""

    datasource: Literal["python_docs", "js_docs"] = Field(
        description='Given the user request or query chose the most relevant data source for a response'
    )

class SelectRoute():
    def __init__(self):
        self.model_name = 'gpt-4o-mini'

    def routeModel(self):
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=self.model_name, temperature=0.7, max_tokens=300)
        return model.with_structured_output(RouteModel)

    def getRoutePrompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages(
            [
                ("system", """You are an expert at routing a user question to the appropriate data source.Based on the 
                specifics of the question, route it to the relevant data source."""),
                ("human", "Question: {question}")
            ]
        )

    def get_route(self, question):
        router = self.routeModel()
        router_prompt = self.getRoutePrompt()
        router_chain = router_prompt | router
        route = router_chain.invoke(question)
        return route.datasource.lower()

class chatBot():
    def __init__(self):
        self.model_name = 'gpt-4o-mini'
        self.py_source = 'C:\\Users\\ajkay\\IdeaProjects\\ai-projects\\dataset\\PromptTemplates_Python.pdf'
        self.py_db_location = 'C:\\Users\\ajkay\\IdeaProjects\\ai-projects\\vectorstore\\python'
        self.js_source = 'C:\\Users\\ajkay\\IdeaProjects\\ai-projects\\dataset\\PromptTemplates_JS.pdf'
        self.js_db_location = 'C:\\Users\\ajkay\\IdeaProjects\\ai-projects\\vectorstore\\js'

    def retrieve_context(self, question):
        retriever = RetrieverApp()
        router = SelectRoute()
        route = router.get_route(question)

        if 'python_docs' in route:
            py_retriever = retriever.GetRetriever(file_path=self.py_source, db_location=self.py_db_location)
            return py_retriever.invoke(question)
        else:
            js_retriever = retriever.GetRetriever(file_path=self.js_source, db_location=self.js_db_location)
            return js_retriever.invoke(question)

    def join_doc(self, docs):
        return '\n'.join(doc.page_content.strip() for doc in docs)

    def getUserPrompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages(
            [("system", "Answer the question using the given contest"),
             ("human", "Context: {context}"),
             ("human", "Question: {question}")
             ]
        )

    def chatModel(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model_name, temperature=0.7, max_tokens=500)

    def getChain(self):
        from operator import itemgetter
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda
        prompt = self.getUserPrompt()
        llm_model = self.chatModel()

        return (
                {
                    "context":itemgetter("question") | RunnableLambda(self.retrieve_context) | RunnableLambda(self.join_doc),
                    "question": itemgetter("question")
                }
            | prompt
            | llm_model
            | StrOutputParser()
        )

    def rag_response(self, question):
        chain = self.getChain()
        return chain.invoke({"question": question})

if __name__=="__main__":
    question = """Why doesn't the following code work:

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
            prompt.invoke("french")
            """
    with trace(project_name='Technical Bot', input={"question": question}, name='Rag for Technical question') as rt:
        app = chatBot()
        response = app.rag_response(question)
        rt.end(outputs={"answer": response})

