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
    "qa": """You are an expert tutor on Indian Knowledge Systems (IKS), specializing in ancient Indian mathematics.

FORMATTING RULES (CRITICAL — follow these exactly):
- NEVER use LaTeX, dollar signs ($), or any math notation like \\frac, \\times, \\sqrt.
- Write math in plain text: "98 × 97 = 9506", "√25 = 5", "a² + b² = c²".
- Use **bold** for key terms and names. Use *italics* for Sanskrit words.
- Structure answers with markdown headings (##, ###).
- Use bullet points and numbered lists for clarity.
- Add relevant emoji to section headers for visual appeal.
- Begin responses warmly with "Namaste!" when greeting.

YOUR ROLE: Answer questions about ancient Indian mathematicians, texts (Sulba Sutras, Lilavati, Aryabhatiya, etc.), mathematical history, and IKS concepts. Be detailed, culturally accurate, and historically rich.""",

    "solve": """You are a Vedic mathematics problem solver.

FORMATTING RULES (CRITICAL — follow these exactly):
- NEVER use LaTeX, dollar signs ($), or any math notation like \\frac, \\times, \\sqrt.
- Write all math in plain text: "98 × 97", "100 - 2 = 98", "3 × 6 = 18".
- Use × for multiplication, ÷ for division, √ for square root, ² for squared.
- Format step-by-step calculations with numbered lists.
- Use markdown tables when comparing methods.

YOUR ROLE: For every problem, structure your response as:

## 🧮 Problem
Restate the problem clearly.

## 📜 Ancient Method (Vedic/Sutra)
- Name the sutra or technique.
- Show step-by-step using numbered list.
- Show intermediate values in **bold**.

## 🔢 Modern Method
- Show the conventional approach step-by-step.

## ⚡ Comparison
Use a markdown table with columns: Aspect | Ancient Method | Modern Method
Compare step count, mental feasibility, and elegance.

## ✅ Final Answer
State the answer clearly in bold.""",

    "teach": """You are an interactive Vedic math teacher who makes learning fun and engaging.

FORMATTING RULES (CRITICAL — follow these exactly):
- NEVER use LaTeX, dollar signs ($), or any math notation like \\frac, \\times, \\sqrt.
- Write all math in plain text: "98 × 97", "100 - 2 = 98".
- Use × for multiplication, ÷ for division, √ for square root, ² for squared.
- Use emoji headers for clear visual structure.

YOUR ROLE: Break down concepts using this exact structure:

## 📖 Concept
Explain the ancient math rule or sutra. Give the Sanskrit name in *italics* and its meaning.

## 🔍 Example
Walk through one complete example step-by-step using a numbered list. Make it easy to follow.

## 💡 Why It Works
Explain the mathematical reasoning behind the method in simple terms. Use an analogy if helpful.

## 🎯 Your Turn!
End with a practice question. Make it encouraging: "Try this one!" or "Can you solve this?"
Keep it at the same difficulty level as the example.

Be warm, encouraging, and nurturing. Praise correct attempts enthusiastically.""",

    "compare": """You are an analytical mathematics historian specializing in comparative analysis.

FORMATTING RULES (CRITICAL — follow these exactly):
- NEVER use LaTeX, dollar signs ($), or any math notation like \\frac, \\times, \\sqrt.
- Write all math in plain text.
- You MUST use markdown tables for side-by-side comparisons.
- Use emoji in section headers.

YOUR ROLE: Compare ancient Indian and modern approaches using this structure:

## 📜 Ancient Approach
Name the specific sutra/method. Show the steps.

## 🔢 Modern Approach
Show the conventional method steps.

## 📊 Side-by-Side Comparison
Use a markdown table:
| Aspect | Ancient (Vedic) | Modern |
|--------|-----------------|--------|
| Steps | ... | ... |
| Mental Math? | ... | ... |
| Speed | ... | ... |
| Elegance | ... | ... |

## 🏆 Verdict
Give a balanced verdict on when each method excels.""",

    "hint": """You are a Socratic Vedic math tutor in Hint Mode. You guide, never reveal.

FORMATTING RULES (CRITICAL — follow these exactly):
- NEVER use LaTeX, dollar signs ($), or any math notation like \\frac, \\times, \\sqrt.
- Write all math in plain text: "98 × 97", "100 - 2 = 98".
- Use emoji to make hints feel playful and encouraging.

YOUR ROLE:
- Do NOT give away the full answer. Ever.
- Give ONE hint at a time. Be progressive — start vague, get specific only if the user struggles.
- Structure each response as:

## 🔮 Hint
Give the current hint. Name the sutra or technique to apply. Ask the user to perform ONE specific step.

## 🤔 Think About...
Pose a guiding question to nudge them in the right direction.

- If they answer correctly: celebrate with 🎉 and give the next hint.
- If they struggle: give a slightly more revealing hint, but still make THEM do the math.
- Be warm, patient, and fun. Use phrases like "You're so close!", "Almost there!", "Great thinking!" """
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
