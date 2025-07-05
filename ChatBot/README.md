# 🧠 Chatbot Assistant with SQLite persistent Memory

This project is an AI-powered chatbot that generates SQL queries based on user questions and retains memory across conversations using **LangChain**, **LangGraph**, **SQLite**, and **OpenAI's GPT models**.

It supports both a **CLI interface** and a **Streamlit-based web interface**, with persistent chat history per user.

---

## 🔧 Features

- ✅ User-focused AI assistant
- ✅ Persistent memory using SQLite
- ✅ Powered by OpenAI (GPT-4o / GPT-4o-mini)
- ✅ Built with LangChain + LangGraph
- ✅ CLI interface for terminal usage
- ✅ Streamlit UI for browser-based interaction
- ✅ System prompts and memory reset support

---

## 📁 ChatBot

```bash
.
├── app/conversational_bot_with_memory.py        # Main chatbot logic (LangGraph, OpenAI, SQLite)
├── app/cli.py         # CLI script to interact with the chatbot
├── app/chat_ui.py      # Streamlit web app interface
├── app/chat_memory.db      # SQLite database (auto-generated)
└── README.md           # You are here!
