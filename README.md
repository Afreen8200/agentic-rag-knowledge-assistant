
# Agentic RAG Knowledge Assistant

An AI-powered knowledge assistant that answers questions using the GitLab Communication Handbook.

## Features
- Document ingestion and section-aware chunking
- Hybrid retrieval using BM25 and semantic search
- ChromaDB vector database
- Local Llama 3.2 3B inference through Ollama
- LangGraph agent with tool calling
- Extractive guardrail for documented procedures
- FastAPI endpoint for question answering
- Basic RAG evaluation

## Technologies
Python, LangGraph, LangChain, ChromaDB, Hugging Face, BM25, Ollama, Llama 3.2 3B, FastAPI.

## API
POST /ask

Example request:
{
  "question": "What should employees do if Slack is down?"
}

## Data Source
GitLab Communication Handbook:
https://handbook.gitlab.com/handbook/communication/

## Status
Working local prototype. Docker deployment is planned.
