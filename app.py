"""
RIO — AI Voice Call Assistant
=============================
Turn-based voice call:  Mic -> Whisper (Groq) -> LLM (Groq) -> TTS -> Speaker

Streamlit cannot do true full-duplex streaming audio, so RIO uses a stable
push-to-talk flow: you record a turn, RIO answers out loud, you record again.
RIO's own audio is never sent back into the speech-to-text step.
"""

import asyncio
import base64
import html
import io
import re
import time

import groq
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

# ----------------------------------------------------------------------------
# CONFIGURATION — change models/voice here
# ----------------------------------------------------------------------------
# NOTE: llama-3.1-8b-instant was shut down for free/developer tiers on
# 16 Aug 2026. openai/gpt-oss-20b is Groq's current fast production model.
GROQ_MODEL = "openai/gpt-oss-20b"
STT_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]  # tried in order
STT_LANGUAGE = "en"  # set to None for auto-detect
TTS_VOICE = "en-US-AriaNeural"  # any edge-tts voice name
MAX_HISTORY_MESSAGES = 12  # recent messages sent to Groq (plus system prompt)
MAX_REPLY_TOKENS = 1024  # gpt-oss reasoning tokens count toward this limit
MIN_AUDIO_BYTES = 2000  # anything smaller is treated as empty audio

GREETING = "Hi, I'm RIO. Press the microphone, tell me what's on your mind, and I'll answer."

SYSTEM_PROMPT = (
    "You are RIO, a friendly, intelligent and confident personal AI voice assistant. "
    "You are speaking out loud on a voice call, so: keep answers short (usually one to "
    "three sentences), use natural spoken language, never use markdown, lists, emojis "
    "or code blocks, and avoid repeating the same phrases. Remember the earlier parts "
    "of the conversation and answer follow-up questions naturally. If you need more "
    "detail from the user, ask one brief question."
)

# Call states
IDLE, LISTENING, THINKING, SPEAKING, ERROR = "IDLE", "LISTENING", "THINKING", "SPEAKING", "ERROR"
STATUS_TEXT = {
    IDLE: "Ready to talk",
    LISTENING: "Listening… press the mic and speak",
    THINKING: "Thinking…",
    SPEAKING: "RIO is speaking…",
    ERROR: "Something went wrong",
}


class RioError(Exception):
    """An error whose message is safe to show to the user."""


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "state": IDLE,
        "call_active": False,
        "call_start_ms": 0,
        "mic_key": 0,  # changing it resets the recorder after every turn
        "flash_audio": None,  # RIO's reply, played once on next render
        "flash_error": None,
        "last_audio": None,  # kept for the manual replay player
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ----------------------------------------------------------------------------
# GROQ CLIENT + ERROR MAPPING
# ----------------------------------------------------------------------------
def get_groq_client():
    """Create the Groq client from Streamlit Secrets (never hard-coded)."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None
    if not api_key or "your_groq_api_key" in str(api_key):
        return None
    return Groq(api_key=api_key)


def friendly_error(exc: Exception) -> str:
    """Translate any exception into a safe, friendly message (no secrets)."""
    if isinstance(exc, RioError):
        return str(exc)
    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return "RIO couldn't connect right now. Please check your API configuration and try again."
    if isinstance(exc, groq.NotFoundError):
        return "The configured Groq model isn't available. Update GROQ_MODEL in app.py."
    if isinstance(exc, groq.RateLimitError):
        return "RIO is receiving too many requests. Please wait a moment and try again."
    if isinstance(exc, groq.APIConnectionError):
        return "RIO lost the network connection. Please check your internet and try again."
    if isinstance(exc, groq.APIStatusError):
        return "Groq returned an error. Please try again in a moment."
    return "Something unexpected happened. Please try again."


# ----------------------------------------------------------------------------
# SPEECH -> TEXT
# ----------------------------------------------------------------------------
def speech_to_text(client: Groq, audio_bytes: bytes) -> str:
    if not audio_bytes or len(audio_bytes) < MIN_AUDIO_BYTES:
        raise RioError("I didn't hear anything. Please try speaking again.")
    last_exc = None
    for model in STT_MODELS:
        try:
            kwargs = {"model": model, "file": ("speech.wav", audio_bytes), "response_format": "json"}
            if STT_LANGUAGE:
                kwargs["language"] = STT_LANGUAGE
            result = client.audio.transcriptions.create(**kwargs)
            return (result.text or "").strip()
        except (groq.NotFoundError, groq.BadRequestError) as exc:
            last_exc = exc  # model unavailable -> try the next one
    raise last_exc or RioError("Speech recognition failed.")


# ----------------------------------------------------------------------------
# LLM RESPONSE
# ----------------------------------------------------------------------------
def generate_ai_response(client: Groq, messages: list) -> str:
    # Send the system prompt + only the most recent turns (speed / token use).
    trimmed = [messages[0]] + messages[1:][-MAX_HISTORY_MESSAGES:]
    params = dict(model=GROQ_MODEL, messages=trimmed, temperature=0.7, max_completion_tokens=MAX_REPLY_TOKENS)
    try:
        completion = client.chat.completions.create(reasoning_effort="low", **params)
    except groq.BadRequestError:
        completion = client.chat.completions.create(**params)  # model w/o reasoning_effort
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise RioError("RIO couldn't think of a reply. Please try again.")
    return text


# ----------------------------------------------------------------------------
# TEXT -> SPEECH  (edge-tts neural voice, gTTS fallback; never crashes the app)
# ----------------------------------------------------------------------------
def _clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_`#>~|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def _edge_tts(text: str) -> bytes:
    import edge_tts

    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, TTS_VOICE, rate="+5%").stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return bytes(audio)


def text_to_speech(text: str):
    """Return MP3 bytes, or None if every TTS engine fails."""
    text = _clean_for_speech(text)
    if not text:
        return None
    try:
        audio = asyncio.run(_edge_tts(text))
        if audio:
            return audio
    except Exception:
        pass
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# ONE FULL TURN
# ----------------------------------------------------------------------------
def process_turn(client: Groq, audio_bytes: bytes, speaker_on: bool):
    """User audio -> text -> Groq -> text -> speech. Returns reply MP3 bytes or None."""
    user_text = speech_to_text(client, audio_bytes)
    if not user_text:
        raise RioError("I couldn't make out any words. Please try again.")
    messages = st.session_state.messages
    messages.append({"role": "user", "content": user_text})
    try:
        reply = generate_ai_response(client, messages)
    except Exception:
        messages.pop()  # keep history consistent if the request failed
        raise
    messages.append({"role": "assistant", "content": reply})
    return text_to_speech(reply) if speaker_on else None


# ----------------------------------------------------------------------------
# UI: STYLING
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility:hidden;}
.stApp {background: radial-gradient(1200px 700px at 50% -10%, #1b1d3a 0%, #0a0b12 55%, #050509 100%); color:#f2f4ff;}
.block-container {max-width: 720px; padding-top: 1.5rem; padding-bottom: 3rem;}
.rio-title {text-align:center; font-size:3rem; font-weight:800; letter-spacing:.45em; margin:0 0 0 .45em;
  background:linear-gradient(90deg,#67e8f9,#818cf8,#c084fc); -webkit-background-clip:text; color:transparent;}
.rio-sub {text-align:center; color:#a9b0d0; margin:.2rem 0 .6rem; font-size:1rem; letter-spacing:.08em;}
.rio-status {text-align:center; font-size:1.25rem; font-weight:600; margin:.2rem 0 1rem;}
.rio-card {background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09); border-radius:18px; padding:14px 18px; margin-top:12px;}
.rio-chip {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.82rem; border:1px solid rgba(255,255,255,.15); color:#d6dbf5; margin:0 4px;}
.rio-msg {margin:.45rem 0; line-height:1.45;}
.rio-who {font-weight:700; font-size:.78rem; letter-spacing:.1em; text-transform:uppercase;}
.rio-you {color:#93c5fd;} .rio-rio {color:#c4b5fd;}
.rio-error {background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.5); color:#fecaca; border-radius:14px; padding:12px 16px; margin:8px 0;}
.stButton > button {border-radius:999px; font-weight:700; padding:.7rem 1.6rem; border:1px solid rgba(255,255,255,.15);
  background:rgba(255,255,255,.06); color:#fff; transition:all .2s ease; width:100%;}
.stButton > button:hover {transform:translateY(-2px); border-color:#818cf8; box-shadow:0 8px 24px rgba(129,140,248,.35); color:#fff;}
.st-key-start_btn button {background:linear-gradient(90deg,#10b981,#06b6d4); font-size:1.2rem; padding:1rem 2rem; border:none;}
.st-key-end_btn button {background:linear-gradient(90deg,#ef4444,#f97316); font-size:1.2rem; padding:1rem 2rem; border:none;}
[data-testid="stAudioInput"] {border-radius:18px;}
</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# UI: AVATAR (CSS/JS orb rendered in an iframe so it can react to audio events)
# ----------------------------------------------------------------------------
AVATAR_TEMPLATE = """
<style>
 html,body{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;color:#eef0ff;overflow:hidden}
 #wrap{display:flex;flex-direction:column;align-items:center;gap:10px}
 #timer{font-size:.9rem;color:#a9b0d0;height:1.1rem;letter-spacing:.1em}
 #stage{position:relative;width:260px;height:260px;display:flex;align-items:center;justify-content:center}
 .ring{position:absolute;border-radius:50%;border:2px solid rgba(129,140,248,.35);opacity:.6}
 .r1{width:150px;height:150px}.r2{width:200px;height:200px;border-color:rgba(103,232,249,.28)}
 .r3{width:250px;height:250px;border-color:rgba(192,132,252,.22)}
 #core{width:110px;height:110px;border-radius:50%;
   background:radial-gradient(circle at 35% 30%,#ffffff 0%,#a5f3fc 18%,#818cf8 55%,#4c1d95 100%);
   box-shadow:0 0 40px rgba(129,140,248,.6),0 0 90px rgba(103,232,249,.25);animation:breathe 4s ease-in-out infinite;position:relative;z-index:2}
 #waves{position:absolute;z-index:3;display:flex;gap:5px;align-items:center;height:60px;opacity:0;transition:opacity .3s}
 #waves i{display:block;width:6px;height:14px;border-radius:4px;background:#fff;opacity:.95}
 #mic{position:absolute;z-index:3;font-size:30px;opacity:0;transition:opacity .3s}
 #orbit{position:absolute;width:210px;height:210px;opacity:0;transition:opacity .3s}
 #orbit b{position:absolute;width:12px;height:12px;border-radius:50%;background:#67e8f9;box-shadow:0 0 12px #67e8f9}
 #orbit b:nth-child(1){top:0;left:99px}#orbit b:nth-child(2){bottom:14px;left:20px;background:#c084fc}#orbit b:nth-child(3){bottom:14px;right:20px;background:#818cf8}
 #pill{padding:5px 16px;border-radius:999px;font-weight:700;letter-spacing:.18em;font-size:.78rem;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.06)}
 #label{font-size:1rem;color:#cdd3f5;min-height:1.3rem;text-align:center}
 @keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}
 @keyframes ripple{0%{transform:scale(.85);opacity:.7}100%{transform:scale(1.25);opacity:0}}
 @keyframes bars{0%,100%{height:12px}50%{height:56px}}
 @keyframes spin{to{transform:rotate(360deg)}}
 @keyframes micpulse{0%,100%{transform:scale(1)}50%{transform:scale(1.25)}}
 .LISTENING #mic{opacity:1;animation:micpulse 1.3s infinite}
 .LISTENING .ring{border-color:rgba(52,211,153,.55);animation:ripple 2s infinite}
 .LISTENING .r2{animation-delay:.5s}.LISTENING .r3{animation-delay:1s}
 .LISTENING #core{box-shadow:0 0 50px rgba(52,211,153,.6)}
 .THINKING #orbit{opacity:1;animation:spin 1.6s linear infinite}
 .THINKING #core{animation:breathe 1s ease-in-out infinite}
 .SPEAKING #waves{opacity:1}
 .SPEAKING #waves i{animation:bars .9s ease-in-out infinite}
 .SPEAKING #waves i:nth-child(2){animation-delay:.1s}.SPEAKING #waves i:nth-child(3){animation-delay:.2s}
 .SPEAKING #waves i:nth-child(4){animation-delay:.3s}.SPEAKING #waves i:nth-child(5){animation-delay:.4s}
 .SPEAKING .ring{animation:ripple 1.4s infinite}.SPEAKING .r2{animation-delay:.35s}.SPEAKING .r3{animation-delay:.7s}
 .SPEAKING #core{box-shadow:0 0 60px rgba(192,132,252,.9),0 0 120px rgba(103,232,249,.45);animation:breathe .8s ease-in-out infinite}
 .ERROR #core{background:radial-gradient(circle at 35% 30%,#fff,#fecaca 20%,#ef4444 60%,#7f1d1d);box-shadow:0 0 50px rgba(239,68,68,.6)}
 .ERROR .ring{border-color:rgba(239,68,68,.4)}
 .IDLE .ring{animation:ripple 4s infinite;opacity:.35}
</style>
<div id="wrap">
  <div id="timer"></div>
  <div id="stage" class="__STATE__">
    <div class="ring r3"></div><div class="ring r2"></div><div class="ring r1"></div>
    <div id="orbit"><b></b><b></b><b></b></div>
    <div id="core"></div>
    <div id="waves"><i></i><i></i><i></i><i></i><i></i></div>
    <div id="mic">🎙️</div>
  </div>
  <div id="pill">__STATE__</div>
  <div id="label"></div>
</div>
<script>
 const stage=document.getElementById('stage'), pill=document.getElementById('pill'), label=document.getElementById('label');
 const LABELS=__LABELS__;
 function setState(s,text){stage.className=s;pill.textContent=s;label.textContent=text||LABELS[s];}
 setState("__STATE__", __CUSTOM_LABEL__);
 const START=__START__, timer=document.getElementById('timer');
 if(START>0){const t=()=>{const s=Math.floor((Date.now()-START)/1000);
   timer.textContent="● "+String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");};t();setInterval(t,1000);}
 const B64="__AUDIO__";
 if(B64){
   const a=new Audio("data:audio/mpeg;base64,"+B64);
   a.onended=()=>setState("LISTENING","Your turn — press the mic and speak");
   a.onerror=()=>setState("LISTENING","Audio couldn't play — use the replay player below");
   a.play().catch(()=>setState("LISTENING","Browser blocked autoplay — use the replay player below"));
 }
</script>
"""


def render_avatar(placeholder, state: str, audio_bytes=None, label=None):
    import json

    page = (
        AVATAR_TEMPLATE.replace("__LABELS__", json.dumps(STATUS_TEXT))
        .replace("__CUSTOM_LABEL__", json.dumps(label))
        .replace("__STATE__", state)
        .replace("__START__", str(int(st.session_state.call_start_ms)))
        .replace("__AUDIO__", base64.b64encode(audio_bytes).decode() if audio_bytes else "")
    )
    with placeholder.container():
        components.html(page, height=430)


# ----------------------------------------------------------------------------
# UI: CALL CONTROLS
# ----------------------------------------------------------------------------
def render_call_controls(client: Groq):
    if not st.session_state.call_active:
        if st.button("☎  Start Call", key="start_btn"):
            st.session_state.call_active = True
            st.session_state.call_start_ms = int(time.time() * 1000)
            st.session_state.state = LISTENING
            with st.spinner("RIO is picking up…"):
                audio = text_to_speech(GREETING) if not st.session_state.get("speaker_muted") else None
            st.session_state.messages.append({"role": "assistant", "content": GREETING})
            st.session_state.flash_audio = audio
            st.rerun()
    else:
        if st.button("📵  End Call", key="end_btn"):
            st.session_state.call_active = False
            st.session_state.call_start_ms = 0
            st.session_state.state = IDLE
            st.session_state.mic_key += 1
            st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.toggle("Mute mic", key="mic_muted")
    c2.toggle("Mute speaker", key="speaker_muted")
    if c3.button("Clear conversation", key="clear_btn"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.last_audio = None
        st.rerun()

    mic = "Off" if not st.session_state.call_active else ("Muted" if st.session_state.mic_muted else "On")
    spk = "Muted" if st.session_state.speaker_muted else "On"
    st.markdown(
        f'<div style="text-align:center;margin:6px 0 10px">'
        f'<span class="rio-chip">Microphone: {mic}</span><span class="rio-chip">Speaker: {spk}</span></div>',
        unsafe_allow_html=True,
    )


def render_conversation():
    history = [m for m in st.session_state.messages if m["role"] != "system"][-8:]
    if not history:
        return
    rows = []
    for m in history:
        who, css = ("You", "rio-you") if m["role"] == "user" else ("RIO", "rio-rio")
        rows.append(
            f'<div class="rio-msg"><div class="rio-who {css}">{who}</div>{html.escape(m["content"])}</div>'
        )
    st.markdown(f'<div class="rio-card">{"".join(rows)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="RIO — AI Voice Call Assistant", page_icon="🎙️", layout="centered")
    initialize_session_state()
    inject_css()

    st.markdown('<div class="rio-title">RIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="rio-sub">Your Personal AI Voice Assistant</div>', unsafe_allow_html=True)

    client = get_groq_client()
    if client is None:
        st.markdown(
            '<div class="rio-error"><b>Groq API key not found.</b><br>Add <code>GROQ_API_KEY</code> to '
            "<code>.streamlit/secrets.toml</code> (local) or Streamlit Cloud → Settings → Secrets. "
            "See the README.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # One-shot items from the previous run (RIO's reply audio / an error message).
    audio = st.session_state.flash_audio
    error = st.session_state.flash_error
    st.session_state.flash_audio = None
    st.session_state.flash_error = None

    avatar_ph = st.empty()
    status_ph = st.empty()

    if error:
        view_state, label = ERROR, error
    elif audio:
        view_state, label = SPEAKING, None
        st.session_state.last_audio = audio
    elif st.session_state.call_active:
        view_state, label = LISTENING, None
    else:
        view_state, label = IDLE, None
    render_avatar(avatar_ph, view_state, audio, label)
    if error:
        st.markdown(f'<div class="rio-error">{html.escape(error)}</div>', unsafe_allow_html=True)

    render_call_controls(client)

    # --- Microphone (turn-based push-to-talk) --------------------------------
    if st.session_state.call_active and not st.session_state.mic_muted:
        recording = st.audio_input(
            "Press the mic, speak, then press stop — RIO will answer",
            key=f"mic_{st.session_state.mic_key}",
        )
        if recording is not None:
            render_avatar(avatar_ph, THINKING)  # show THINKING while we work
            try:
                reply_audio = process_turn(client, recording.getvalue(), not st.session_state.speaker_muted)
                st.session_state.flash_audio = reply_audio
            except Exception as exc:  # friendly message only — never leak keys
                st.session_state.flash_error = friendly_error(exc)
            st.session_state.mic_key += 1  # fresh recorder; old audio is never reprocessed
            st.rerun()
    elif st.session_state.call_active:
        st.info("Microphone is muted. Turn off “Mute mic” to talk to RIO.")

    render_conversation()

    if st.session_state.last_audio:
        with st.expander("🔁 Replay RIO's last answer"):
            st.audio(st.session_state.last_audio, format="audio/mpeg")


main()
