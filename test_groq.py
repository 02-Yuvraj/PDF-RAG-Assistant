
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

# Connect to the Groq language model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Send a simple test message
response = llm.invoke(
    "Reply with exactly: Groq connection successful"
)

print("Groq response:", response.content)
