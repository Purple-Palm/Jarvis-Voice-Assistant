# Interfaces with Ollama (fotiecodes/jarvis) for LLM queries

import ollama
from ollama import chat

conversation = [
    {"role": "user", "content": "Hello, how are you?"}
]
reply = chat(model='fotiecodes/jarvis:latest', messages=conversation)
print(reply.message.content)
 
