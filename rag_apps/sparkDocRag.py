import os
from langsmith import trace
from dotenv import load_dotenv
load_dotenv()


#### Initializing environment variable(s)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

### Data load for document url
class dataload():
    """This class loads data in vector data and creating a retriever.
    The document splitting is based on semantics."""
    def __init__(self):
        self.url = 'https://spark.apache.org/docs/latest/sql-performance-tuning.html'
        self.db_path = "./vectorstore/spark__performance.db"
        self.embedding_model_name = "text-embedding-3-small"
        self.collection_name = "spark_doc_rag"

    def web_loader(self):
        import bs4
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(web_path=self.url,
                           bs_kwargs={"parse_only": bs4.SoupStrainer(id="content")})
        return loader.load()

    def doc_splits(self, document):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        split_criteria = RecursiveCharacterTextSplitter(chunk_size=300,
                                                  chunk_overlap=0,
                                                  separators=['\n\n\n\n','\n\n\n','\n\n', '\n'])
        return split_criteria.split_documents(document)

    def create_vectorstore(self, docs):
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=self.db_path,
            collection_name=self.collection_name
        )
        print("db created")

        vector_store.reset_collection()
        print("collection reset completed")
        vector_store.add_documents(documents=docs)
        print("documents added")
        return vector_store

    def create_retriever(self):
        documents = self.web_loader()
        split = self.doc_splits(documents)
        vector_store = self.create_vectorstore(split)
        return vector_store.as_retriever(search_kwargs={'k': 3})

class docApp():
    """This class create a rag chat app that answer user question based
    on the document stored in the vector database."""
    def __init__(self):
        self.model = "gpt-4o-mini"

    def chat_model(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model, max_tokens=300)

    def chat_prompts(self):
        from langchain_core.prompts import ChatPromptTemplate
        prompts = ChatPromptTemplate(
            [
                ("system", 'template'),
                ("human", "Context: {context}"),
                ("human", "Question: {question}")
            ]
        )

        return prompts

    def getContext(self, question):
        load = dataload()
        retrieve = load.create_retriever()
        return retrieve.invoke(question)


    def join_retrieved_doc(self, docs):
        return "\n".join(doc.page_content.strip() for doc in docs)

    def rag_chain(self ):
        from operator import itemgetter
        from langchain_core.runnables import RunnableLambda
        from langchain_core.output_parsers import StrOutputParser
        get_prompts = self.chat_prompts()
        llm = self.chat_model()

        return (
            {
                "context": itemgetter("query") | RunnableLambda(self.getContext) | RunnableLambda(self.join_retrieved_doc),
                "question": itemgetter("query")
            } | get_prompts | llm | StrOutputParser()
        )

    def get_cgat_response(self, question):
        chain = self.rag_chain()
        return chain.invoke({"query": question})

if __name__ == "__main__":
    question = 'What is AQE?'
    with trace(project_name='spark doc app', inputs={'question': question}, name='') as rt:
        app = docApp()
        response = app.get_cgat_response(question=question)
        rt.end(outputs={"answer": response})
