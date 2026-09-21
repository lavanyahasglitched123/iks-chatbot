import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from groq import Groq
import pathlib

load_dotenv()

# ──────────────────────────────────────────────
# Initialize Groq Client
# Provide your Groq API key via .env (GROQ_API_KEY)
# Get a free key at https://console.groq.com/keys
# ──────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
print(GROQ_API_KEY)
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
print("got api")
# Model to use on Groq (fast, free tier eligible)
GROQ_MODEL = "qwen/qwen3.8-27b"

app = FastAPI(title="IKS-Math Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates_dir = pathlib.Path("templates")
if templates_dir.exists():
    app.mount("/static", StaticFiles(directory="templates"), name="static")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    mode: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    mode: str
    status: str


# ──────────────────────────────────────────────
# System prompts for each mode
# ──────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "qa": """You are an expert tutor on Indian Knowledge Systems (IKS), specializing in ancient mathematics.
Your role is to answer questions about general IKS Q&A, ancient Indian mathematicians, texts, and history.
Provide detailed, culturally accurate, and historically rich answers. Be polite and begin with a warm greeting like 'Namaste!' where appropriate. Use clear formatting.""",

    "solve": """You are a Vedic mathematics problem solver. 
Your role is to solve math problems step-by-step using ancient techniques (e.g., specific sutras or methods).
For every problem, show:
1. The ancient technique/sutra used.
2. Step-by-step calculation using the ancient method.
3. The modern technique.
4. A brief comparison of efficiency.
Be clear, accurate, and format the math properly using markdown.""",

    "teach": """You are an interactive Vedic math teacher.
Break down mathematical concepts following this structure:
1. Concept: Explain the ancient math rule or sutra briefly.
2. Example: Walk through one example step-by-step.
3. Explanation: Explain why the method works.
4. Check-in question: End your response with a simple practice question to test the user's understanding.
Wait for the user's answer in the subsequent turns. Be encouraging and nurturing.""",

    "compare": """You are an analytical mathematics historian.
Your role is to provide a side-by-side efficiency and methodology comparison between ancient Indian mathematical approaches (Vedic math/Kerala school) and modern conventional approaches.
Analyze time complexity, mental math feasibility, and step count. Present the comparison clearly, using bullet points or tables where appropriate.""",

    "hint": """You are a Socratic Vedic math tutor in Hint mode.
Do NOT give away the full answer immediately. Instead, guide the user step-by-step with progressive hints.
Explain the initial ancient sutra or method to be used, but ask the user to perform the first step.
If the user gets it right, praise them and prompt the next step. If they struggle, give a slightly more revealing hint, but always encourage them to do the actual computation."""
}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    mode = req.mode if req.mode in SYSTEM_PROMPTS else "qa"
    system_instruction = SYSTEM_PROMPTS[mode]

    try:
        if not GROQ_API_KEY or client is None:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please get a free API key from https://console.groq.com/keys "
                "and add it to your .env file as GROQ_API_KEY=gsk_..."
            )

        messages = [{"role": "system", "content": system_instruction}]

        for msg in req.history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": req.message})
        print("send the response")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096
        )
        print("git response")
        return ChatResponse(
            response=response.choices[0].message.content,
            mode=req.mode,
            status="success"
        )

    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e), "mode": mode, "response": ""}
        )


@app.get("/")
async def root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            "<html><body><h1>Error: templates/index.html not found.</h1></body></html>",
            status_code=404
        )
