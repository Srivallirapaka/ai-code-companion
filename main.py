import os
import re
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
USE_DEMO = not bool(OPENAI_KEY)

if not USE_DEMO:
    import openai
    openai.api_key = OPENAI_KEY

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ExplainRequest(BaseModel):
    language: str
    topic: str
    level: str = "beginner"


class DebugRequest(BaseModel):
    language: str
    code: str


def extract_json_from_text(text: str):
    """Try to extract a JSON object substring from model output and parse it."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def canned_explain(language, topic, level):
    return {
        "explanation": f"(DEMO) {topic} explained concisely for {level} in {language}.",
        "example_code": f"# Demo example for {language}\nprint('Hello, {language}!')",
        "mini_project_name": f"Demo {topic} mini-project",
        "mini_project_starter_code": "# Demo starter code\n# Build from here..."
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "demo": USE_DEMO})


@app.post("/api/explain")
async def api_explain(req: ExplainRequest):
    prompt = (
        f"You are an expert {req.language} tutor. Explain the concept '{req.topic}' to a {req.level} learner.\n\n"
        "Return a JSON object (no extra text) with these keys:\n"
        "explanation (short paragraph),\n"
        "example_code (string of code in the requested language),\n"
        "mini_project_name (short string),\n"
        "mini_project_starter_code (string),\n"
        "next_steps (short list of strings)\n"
    )

    if USE_DEMO:
        return JSONResponse(content=canned_explain(req.language, req.topic, req.level))

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise, helpful programming tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        text = resp["choices"][0]["message"]["content"]
        data = extract_json_from_text(text)
        if data:
            return JSONResponse(content=data)
        # fallback: give raw
        return JSONResponse(content={"raw": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/debug")
async def api_debug(req: DebugRequest):
    prompt = (
        f"You are a {req.language} expert. The user gave this code:\n\n{req.code}\n\n"
        "Explain any errors or potential bugs; provide a corrected version. Return JSON with keys:\n"
        "errors (list of strings), fixed_code (string), explanation (string).\n"
    )

    if USE_DEMO:
        return JSONResponse(
            content={"errors": ["DEMO: no API key"], "fixed_code": req.code, "explanation": "Demo mode - supply OPENAI_API_KEY to enable real debugging."}
        )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise debugging assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        text = resp["choices"][0]["message"]["content"]
        data = extract_json_from_text(text)
        if data:
            return JSONResponse(content=data)
        return JSONResponse(content={"raw": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))