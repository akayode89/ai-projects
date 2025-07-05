import argparse
from app.conversational_bot_with_memory import ChatApp  # Your main class logic
from dotenv import load_dotenv

load_dotenv()

def run_cli():
    parser = argparse.ArgumentParser(description="Chat with SQL-generating assistant")
    parser.add_argument("user_id", type=str, help="Unique user ID")
    parser.add_argument("message", type=str, help="Message to send to the assistant")
    parser.add_argument("--role", type=str, default="user", help="Role: user or system")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")

    args = parser.parse_args()

    app = ChatApp(model_name=args.model, user_id=args.user_id, user_role=args.role, user_text=args.message)
    result = app.run()
    print(f"Bot: {result['output']}")
