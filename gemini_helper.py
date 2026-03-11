"""
gemini_helper.py  ·  AI Health Guardian
- Auto-retries on 429 rate limit errors
- Falls back through multiple free models automatically
"""

import os, time
import google.generativeai as genai

# ── Load .env ─────────────────────────────────────────────────────────────────
def _load_env():
    for env_path in [".env", "../.env"]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        if k and v:
                            os.environ.setdefault(k, v)
            break

_load_env()
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

# ── Free-tier models in priority order (updated March 2026) ──────────────────
# gemini-2.5-flash-lite  →  fastest, highest free quota
# gemini-2.5-flash       →  fallback 1
# gemini-2.0-flash-lite  →  fallback 2
# gemini-2.0-flash       →  fallback 3
FREE_MODELS = [
    "gemini-2.5-flash-lite-preview-06-17",  # Best free quota, fastest
    "gemini-2.5-flash",                      # Fallback 1
    "gemini-2.0-flash-lite",                 # Fallback 2
    "gemini-2.0-flash",                      # Fallback 3
]

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key and key != "YOUR_GEMINI_API_KEY_HERE" else None

def _configure():
    key = get_api_key()
    if not key:
        return False
    genai.configure(api_key=key)
    return True

SYSTEM_PROMPT = """You are Dr. AI, a friendly and knowledgeable health assistant.
Provide helpful, accurate health information in simple and clear language.
Always remind users to consult a real doctor for diagnosis and treatment.
Keep responses concise, warm, and easy to understand.
Use bullet points when listing multiple items.
Never diagnose definitively — say 'may indicate' or 'could be related to'."""

# ── Core caller with retry + model fallback ───────────────────────────────────
def _call_gemini(prompt: str, system: str = SYSTEM_PROMPT,
                 history: list = None, max_retries: int = 2) -> str:
    """
    Try each free model in order. On 429, wait and retry once, then try next model.
    """
    if not _configure():
        return "⚠️ **API key not set.** Please add your Gemini API key in the sidebar."

    last_error = ""
    for model_name in FREE_MODELS:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system
                )
                if history:
                    chat = model.start_chat(history=history)
                    response = chat.send_message(prompt)
                else:
                    response = model.generate_content(prompt)
                return response.text   # ✅ success

            except Exception as e:
                err_str = str(e)
                last_error = err_str

                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    # Extract retry delay if present
                    wait = 5
                    if "retry_delay" in err_str:
                        try:
                            import re
                            match = re.search(r'seconds:\s*(\d+)', err_str)
                            if match:
                                wait = min(int(match.group(1)), 15)  # cap at 15s
                        except:
                            pass
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue          # retry same model
                    else:
                        break             # try next model
                elif "404" in err_str or "not found" in err_str.lower():
                    # Model not available — try next model immediately
                    break
                else:
                    # Other error — don't retry
                    return f"⚠️ Gemini error: {err_str}"

    # All models exhausted
    if "429" in last_error or "quota" in last_error.lower():
        return (
            "⚠️ **Rate limit reached on all free models.**\n\n"
            "This happens when you've sent too many requests in a short time.\n\n"
            "**Options:**\n"
            "- ⏳ Wait 1-2 minutes and try again\n"
            "- 🔄 The free tier resets every minute/day\n"
            "- 🔑 If this keeps happening, create a new API key at "
            "[aistudio.google.com](https://aistudio.google.com/app/apikey)"
        )
    return f"⚠️ Gemini error: {last_error}"


# ── Public functions ──────────────────────────────────────────────────────────
def chat_with_gemini(user_message: str, history: list = []) -> str:
    # Build Gemini-format history
    gemini_history = []
    for msg in history[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    return _call_gemini(user_message, history=gemini_history)


def get_medicine_info(medicine_name: str) -> str:
    prompt = f"""Provide clear information about the medicine: {medicine_name}

Include these sections with HTML bold labels:
<b>Generic Name:</b> ...
<b>Drug Class:</b> ...
<b>Primary Uses:</b> (bullet points)
<b>Common Side Effects:</b> (bullet points)
<b>Typical Dosage:</b> ...
<b>Important Warnings:</b> ...
<b>Storage:</b> ...

Keep it concise and in simple language. End with a reminder to consult a pharmacist or doctor."""
    result = _call_gemini(prompt, system="You are a pharmacist assistant. Give clear, accurate medicine information.")
    return result


def analyze_symptoms_with_gemini(symptoms: list) -> str:
    prompt = f"""A patient reports these symptoms: {", ".join(symptoms)}

Provide:
1. Possible conditions these symptoms may indicate (list 3-5)
2. Immediate steps the person should take
3. Warning signs that require emergency care
4. General preventive measures

Always recommend consulting a doctor."""
    return _call_gemini(prompt)


def get_health_tips(condition: str) -> str:
    prompt = f"""Give 5 practical daily health tips for someone managing: {condition}
Format as a numbered list. Be specific and actionable. 1-2 sentences each."""
    return _call_gemini(prompt, system="You are a health advisor. Give practical, evidence-based tips.")