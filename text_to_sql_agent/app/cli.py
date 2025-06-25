import argparse
from app.text_to_sql_agent import TextToSQLAgent

def run_cli():
    parser = argparse.ArgumentParser(description="Convert natural language questions into SQL queries.")
    parser.add_argument("question", type=str, help="User question to convert into SQL")

    args = parser.parse_args()
    agent = TextToSQLAgent()
    result = agent.run(args.question)

    print("\n--- Final Output ---")
    print("User Query:", result["user_query"])
    print("SQL Query:", result["sql_query"])
    print("Explanation:", result["query_explanation"])
