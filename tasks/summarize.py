import requests
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langsmith import trace
from newspaper import Article
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def newsArticleSummarizer(url, max_tokens, model_name="gpt-4o-mini"):
    """The function summarizes new article from the given url using OpenAI API"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

    article_url = f"{url}"

    session = requests.Session()

    template = """You are a very good assistant that summarizes online articles.
                    Here's the article you want to summarize.
                    ==================
                    Title: {article_title}

                    {article_text}
                    ==================

                    Write a summary of the previous article.
                """
    try:
        response = session.get(article_url, headers=headers, timeout=10)

        if response.status_code == 200:
            article = Article(article_url)
            article.download()
            article.parse()
            article_title = article.title
            article_text = article.text
            prompt = PromptTemplate.from_template(template)
            llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=max_tokens)
            chain = (
                {
                'article_text': itemgetter('article_text'),
                'article_title': itemgetter("article_title")
                }
                | prompt | llm)
            result =chain.invoke({'article_text': article_text, 'article_title': article_title})
            result = result.content
            return result

        else:
            print(f"Failed to fetch article at {article_url}")
    except Exception as e:
        print(f"Error occurred while fetching article at {article_url}: {e}")

if __name__=="__main__":
    url = 'https://www.artificialintelligence-news.com/news/ai-adoption-matures-deployment-hurdles-remain/'
    max_tokens = 300
    model_name = 'gpt-4o-mini'
    with trace(project_name="News Article Summarization Assistant",
               inputs={"url": url, "max_token": max_tokens, "model_name": model_name},
               name='Summary') as rt:
        summary = newsArticleSummarizer(url=url, max_tokens=max_tokens, model_name=model_name)
        rt.end(outputs={"summary": summary})