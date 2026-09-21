# IKS-Math Tutor Chatbot

A fully functional, production-ready web application for an intelligent Indian Knowledge System (IKS) Mathematics Tutor chatbot.

## Features
- **5 Custom AI Modes:** Q&A, Solve, Teach, Compare, and Hint Mode.
- **Backend:** Python FastAPI backend with Pydantic for validation and `google-genai` SDK integration.
- **Frontend:** Sleek, single-page UI built with Tailwind CSS, Markdown rendering, Lucide icons, and modern design aesthetics (dark slate, amber highlights).

## 🚀 Setup Instructions

1. **Activate Virtual Environment** (Optional but recommended):
```bash
python -m venv venv
# On Windows use:
venv\Scripts\activate
# On macOS/Linux use:
source venv/bin/activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Environment**:
- Rename `.env.example` to `.env`
- Add your Google Gemini API Key: `GEMINI_API_KEY="your_actual_key_here"`

4. **Run the Application**:
```bash
uvicorn main:app --reload
```

5. **Open in Browser**:
Navigate to `http://localhost:8000` to interact with the IKS-Math Tutor!
