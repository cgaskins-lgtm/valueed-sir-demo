import os
import io
import base64
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

# ── Environment ────────────────────────────────────────────────────────────────

load_dotenv()

# Resolve API keys — Streamlit Cloud secrets take priority, then .env
_anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", None) or os.getenv("ANTHROPIC_API_KEY", "")
_openai_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", "")

claude = anthropic.Anthropic(api_key=_anthropic_key)
oai = OpenAI(api_key=_openai_key)

RAG_FOLDER = Path("rag_documents")
REPORTS_FOLDER = Path("generated_reports")
REPORTS_FOLDER.mkdir(exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ValueEd AI SIR Coach",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
/* Hide the recorded audio playback bar — transcription is automatic */
[data-testid="stAudioInput"] audio { display: none !important; }
[data-testid="stAudioInput"] [data-testid="stAudioInputWaveform"] + div { display: none !important; }
/* Make sidebar buttons full width */
section[data-testid="stSidebar"] .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── RAG documents ──────────────────────────────────────────────────────────────

@st.cache_data
def load_rag() -> str:
    docs = []
    for path in sorted(RAG_FOLDER.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = f"[Could not read {path.name}]"
        docs.append(f"\n\n=== {path.name} ===\n{text}")
    return "\n".join(docs)

RAG_CONTEXT = load_rag()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_BASE = f"""You are ValueEd AI Systems — a calm, practical coaching assistant for residential treatment staff.

RULES:
- Reference ONLY the RAG context below for policy, procedures, and resident profiles. Never invent policy.
- Protect confidentiality: use Sam's name for the primary resident in the report; use "Resident #1" and "Resident #2" for all other youth.
- Use observable, behavior-based language only — no diagnosis, blame, or interpretation.
- Keep responses concise and practical. Staff may be stressed.
- Ask ONE clarifying question at a time when information is missing.
- Remind staff this is a training demo and to follow their agency's emergency and mandated reporting procedures.

TONE: Calm, warm, supportive — like an experienced supervisor helping a colleague think clearly under stress.

RAG CONTEXT:
{RAG_CONTEXT}"""

# ── Modes ──────────────────────────────────────────────────────────────────────

MODES = {
    "📝 SIR Writing": {
        "description": "I'll guide you through documenting this incident step by step.",
        "opening": (
            "I'm here to help you write a clear, complete Serious Incident Report. "
            "Tell me what happened in your own words — start wherever feels natural "
            "and we'll organize it into the required sections together."
        ),
        "addendum": """
CURRENT MODE: SIR Writing Assistant

Guide the staff member through a complete, policy-compliant Serious Incident Report.

When they describe the incident, help build these sections:
1. Antecedent — what happened immediately before the incident
2. Description of Incident — observable behaviors only, no interpretation
3. Staff Interventions Attempted — before any restraint was used
4. Physical Restraint — type, duration, justification (if applicable)
5. Outcome — youth's status after the incident
6. Notifications/Follow-Up — who was notified, when, by whom

Ask ONE clarifying question at a time for missing details.
When the report seems complete, offer to generate a formatted draft.
Use "Resident #1" and "Resident #2" for all youth other than Sam.""",
    },
    "✅ Policy Review": {
        "description": "Share your draft SIR and I'll flag any policy or compliance issues.",
        "opening": (
            "I'll review your SIR against YFT policy and Virginia state requirements. "
            "Share what you have — even rough notes — and I'll identify what's working, "
            "what's missing, and what language needs to change."
        ),
        "addendum": """
CURRENT MODE: Policy Review

Review the provided SIR draft for policy compliance. Identify:
- Confidentiality violations (resident names, staff names used incorrectly)
- Missing required SIR elements
- Subjective, interpretive, or diagnostic language — give better alternatives
- Restraint documentation gaps
- Notification gaps or errors
- Blame, judgment, or opinion language

Be specific. Give corrected wording examples where helpful.""",
    },
    "💬 Debrief": {
        "description": "Let's talk through what happened and how to prevent it next time.",
        "opening": (
            "This debrief is a learning conversation — not a review or discipline. "
            "The goal is to understand what happened, recognize any patterns, "
            "and find better ways to support youth next time.\n\n"
            "First — how are you doing after all of that?"
        ),
        "addendum": """
CURRENT MODE: Debrief Coach

Guide a warm, supportive post-incident debrief using this coaching model:
VALIDATE → EXPLORE (briefly) → ADVISE → PLAN

TONE AND LANGUAGE RULES:
- Never use clinical or medical terminology (e.g. do NOT say "cross-contamination" — say "reducing the risk of the situation spreading to other residents")
- Avoid interrogating the staff member. Balance every 1-2 reflective questions with affirmation, observations, and concrete advice.
- Staff should leave feeling supported and equipped — not pressured or blamed.
- If the staff member seems unsure or uncomfortable, offer your own observations rather than pressing for answers.

ALWAYS PROACTIVELY COVER THESE TOPICS — do not wait to be asked:
1. POWER STRUGGLE RECOGNITION — Identify specific moments in the incident where a power struggle developed (e.g. blocking access to the phone, following a youth who asked to be left alone, issuing threats during restraint). Name it clearly and compassionately: "That moment sounds like it may have shifted into a power struggle — that's very common and worth exploring."
2. DE-ESCALATION TRAINING — Reference Handle with Care techniques and Mental Health First Aid strategies from the training documents. Remind staff of specific skills they can use next time. Frame this as a resource, not a criticism.
3. EARLY WARNING SIGNS — Help staff identify what Sam's profile tells us about his known triggers (visits, mother contact, peer conflict) and how those were active in this incident.
4. CONCRETE NEXT STEPS — End every response with at least one specific, actionable recommendation for what to do differently next time.

WHAT TO AVOID:
- Asking more than 2 questions in a row without offering advice or affirmation
- Using clinical, medical, or punitive language
- Leaving staff without clear, practical guidance
- Focusing only on what went wrong — also acknowledge what staff did right""",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def call_claude(mode_name: str, history: list) -> str:
    addendum = MODES[mode_name]["addendum"]
    system = SYSTEM_BASE + addendum

    # Build API messages — exclude greeting placeholders, must start with user
    api_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if not m.get("greeting")
    ]

    if not api_msgs or api_msgs[0]["role"] != "user":
        return "Could you tell me a bit more about what happened? I want to make sure I help you accurately."

    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=system,
        messages=api_msgs,
    )
    return resp.content[0].text


def transcribe_audio(audio_widget) -> str:
    raw = audio_widget.read()
    buf = io.BytesIO(raw)
    buf.name = "recording.wav"
    result = oai.audio.transcriptions.create(model="whisper-1", file=buf)
    return result.text


def generate_audio(text: str) -> bytes:
    resp = oai.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text[:4096],
    )
    return resp.content


def save_report(content: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_FOLDER / f"SIR_{ts}.txt"
    out.write_text(content, encoding="utf-8")
    return out

# ── Session state ──────────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = list(MODES.keys())[0]
if "histories" not in st.session_state:
    st.session_state.histories = {m: [] for m in MODES}
if "voice" not in st.session_state:
    st.session_state.voice = True
if "last_audio" not in st.session_state:
    st.session_state.last_audio = {}  # mode -> audio bytes
if "policy_input_key" not in st.session_state:
    st.session_state.policy_input_key = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏫 ValueEd AI")
    st.caption("SIR Coaching Demo")
    st.divider()

    st.markdown("**Support Mode**")
    for name in MODES:
        btn_type = "primary" if name == st.session_state.mode else "secondary"
        if st.button(name, key=f"btn_{name}", type=btn_type, use_container_width=True):
            st.session_state.mode = name
            st.rerun()

    st.divider()

    # Voice input lives in sidebar so it's always visible — no scrolling needed
    recorded = None
    if st.session_state.mode != "✅ Policy Review":
        st.markdown("**🎤 Voice Input**")
        st.caption("Click to record · click again to stop")
        audio_key = f"audio_{st.session_state.mode}_{len(st.session_state.histories[st.session_state.mode])}"
        recorded = st.audio_input("Record", key=audio_key, label_visibility="collapsed")
        st.divider()

    st.session_state.voice = st.toggle(
        "🔊 Voice Responses",
        value=st.session_state.voice,
        help="AI responses will be read aloud via OpenAI TTS",
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.histories[st.session_state.mode] = []
        st.session_state.last_audio.pop(st.session_state.mode, None)
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ Demo & training use only. "
        "Always follow your agency's emergency and mandated reporting procedures."
    )

# ── Main area ──────────────────────────────────────────────────────────────────

mode = st.session_state.mode
history = st.session_state.histories[mode]
mode_cfg = MODES[mode]

st.markdown(f"# {mode}")
st.caption(mode_cfg["description"])

# Auto-greeting on first visit to each mode
if not history:
    history.append({"role": "assistant", "content": mode_cfg["opening"], "greeting": True})

# ── Paste box for Policy Review (main area) ────────────────────────────────────

pasted_text = None

if mode == "✅ Policy Review":
    st.markdown("**📋 Paste SIR Draft for Review** — paste below, then click Submit:")
    pasted_text = st.text_area(
        "Paste SIR draft here",
        height=180,
        placeholder="Paste the full SIR or incident description here for policy review...",
        label_visibility="collapsed",
        key=f"policy_text_{st.session_state.policy_input_key}",
    )
    submitted = st.button("Submit for Review", type="primary")
    if not submitted:
        pasted_text = None

st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in history:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Persistent audio player — shown below chat when voice response is available
if st.session_state.voice and mode in st.session_state.last_audio:
    st.markdown("---")
    st.caption("🔊 AI Voice Response:")
    st.audio(st.session_state.last_audio[mode], format="audio/mp3", autoplay=True)
    # Attempt autoplay via parent frame (works when browser allows it)
    components.html("""
    <script>
    setTimeout(function() {
        try {
            var audios = window.parent.document.querySelectorAll('audio');
            if (audios.length > 0) {
                audios[audios.length - 1].play();
            }
        } catch(e) {}
    }, 400);
    </script>
    """, height=0)

# ── Text input (pinned to bottom) ─────────────────────────────────────────────

typed = st.chat_input("Or type your message here...")

# ── Process input ──────────────────────────────────────────────────────────────

user_text = None

if typed:
    user_text = typed
elif pasted_text and pasted_text.strip():
    user_text = pasted_text
elif recorded:
    with st.spinner("Transcribing your recording..."):
        try:
            user_text = transcribe_audio(recorded)
        except Exception as e:
            st.error(f"Transcription failed: {e}")

if user_text:
    history.append({"role": "user", "content": user_text})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("ValueEd AI is responding..."):
            try:
                answer = call_claude(mode, history)
            except Exception as e:
                answer = f"I ran into an error: {e}. Please try again."

        st.markdown(answer)

        if st.session_state.voice:
            with st.spinner("Generating voice response..."):
                try:
                    st.session_state.last_audio[mode] = generate_audio(answer)
                except Exception as e:
                    st.caption(f"Voice unavailable: {e}")

    history.append({"role": "assistant", "content": answer})

    # Clear the policy paste box after submission
    if mode == "✅ Policy Review":
        st.session_state.policy_input_key += 1

    # Check if the answer contains a formatted SIR draft — offer to save it
    if "antecedent" in answer.lower() and "outcome" in answer.lower() and mode == "📝 SIR Writing":
        if st.button("💾 Save Draft Report", key=f"save_{len(history)}"):
            saved = save_report(answer)
            st.success(f"Saved to {saved}")

    st.rerun()
