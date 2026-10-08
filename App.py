"""
=====================================================================================
  AURA — Advanced Personalized AI Assistant
  Stack: Streamlit + google-genai (Gemini API)
  Author: Senior AI Engineer
=====================================================================================

Features
--------
1. Dedicated SYSTEM_INSTRUCTION defining Aura's personality & expertise.
2. Secure API key handling (env var -> st.secrets -> masked sidebar input; never hardcoded).
3. Clean, chat-style web UI (st.chat_message / st.chat_input) with streaming responses.
4. Multimodal input: free text AND live camera snapshots (st.camera_input).
5. On-demand image generation / design using Gemini's native image model,
   triggered by a UI toggle or by detecting creation intent in the prompt.

Install
-------
    pip install streamlit google-genai pillow

Run
---
    streamlit run aura_assistant.py

Set your key (recommended, instead of pasting it in the UI):
    export GEMINI_API_KEY="your_api_key_here"      # macOS/Linux
    setx GEMINI_API_KEY "your_api_key_here"         # Windows
=====================================================================================
"""

import os
import io
import re
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types


# =====================================================================================
# 1. PERSONALITY & EXPERTISE — SYSTEM INSTRUCTION
# =====================================================================================
SYSTEM_INSTRUCTION = """
You are AURA (Adaptive Universal Reasoning Assistant) — a highly advanced, personalized
AI assistant created to be a genuine thinking partner, not just a chatbot.

PERSONALITY:
- Warm, articulate, curious, and quietly confident. You explain things clearly without
  being condescending, and you are honest when you are unsure.
- You adapt your tone to the user: concise and technical for experts, patient and
  example-driven for beginners, encouraging for creative work.
- You have a subtle, dry sense of humor but never let it get in the way of being useful.

EXPERTISE:
- Software engineering, system design, and data science (you write clean, correct,
  well-commented code and explain trade-offs).
- Visual analysis — you can look at photos or camera snapshots the user shares and
  describe, diagnose, or reason about what's in them (objects, text, scenes, diagrams,
  handwriting, UI screenshots, etc.).
- Creative and visual design — when asked, you can generate or edit images, and you
  describe your creative choices briefly before/after producing them.
- General knowledge, research synthesis, writing, and strategic/business advice.

BEHAVIOR RULES:
1. Always ground answers in what the user actually asked; state assumptions if the
   request is ambiguous instead of guessing wildly.
2. When given an image (uploaded or captured via camera), analyze it carefully and
   reference specific visual details in your reply.
3. When asked to draw, design, generate, create, sketch, or visualize something, hand
   off to the image-generation capability and give a short, confident caption for the
   result rather than a long preamble.
4. Keep responses well-structured: short paragraphs, bullet points for lists, and code
   blocks for code. Avoid unnecessary filler.
5. Never fabricate facts, sources, or capabilities. If something is outside what you
   can verify, say so plainly.
6. Be safe and responsible: decline harmful requests briefly and suggest a safe
   alternative when reasonable.
"""

# Models (Gemini's current multimodal + native image-generation models)
TEXT_MODEL = "gemini-2.0-flash"
IMAGE_GEN_MODEL = "imagen-3.0-generate-002"  # "Nano Banana" — native image generation/editing

# Keywords used as a fallback trigger for image-generation intent
IMAGE_INTENT_PATTERN = re.compile(
    r"\b(draw|design|generate|create|sketch|paint|render|visuali[sz]e|make).{0,40}"
    r"\b(image|picture|logo|icon|poster|art|illustration|photo|wallpaper|graphic)\b",
    re.IGNORECASE,
)


# =====================================================================================
# 2. SECURE API KEY HANDLING
# =====================================================================================
def get_api_key() -> str | None:
    """
    Resolve the Gemini API key securely, in order of preference:
      1. Environment variable GEMINI_API_KEY (best practice — never hardcode keys).
      2. Streamlit secrets (st.secrets["GEMINI_API_KEY"]) — for deployed apps.
      3. A masked sidebar input, kept only in-session (st.session_state), never logged
         or written to disk.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # no secrets.toml configured — that's fine

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    with st.sidebar:
        st.markdown("### 🔑 API Key")
        st.session_state.api_key = st.text_input(
            "Enter your Gemini API key",
            value=st.session_state.api_key,
            type="password",
            help="Stored only in this browser session's memory — never saved to disk "
                 "or sent anywhere except Google's API.",
        )
    return st.session_state.api_key or None


# =====================================================================================
# 3. GENAI CLIENT & CHAT SESSION (cached in session_state)
# =====================================================================================
def get_client(api_key: str) -> genai.Client:
    if (
        "client" not in st.session_state
        or st.session_state.get("client_key") != api_key
    ):
        st.session_state.client = genai.Client(api_key=api_key)
        st.session_state.client_key = api_key
        # Reset chat session whenever the key changes
        st.session_state.pop("chat", None)
    return st.session_state.client


def get_chat_session(client: genai.Client):
    """Create (once) a persistent multi-turn chat session with our system instruction."""
    if "chat" not in st.session_state:
        st.session_state.chat = client.chats.create(
            model=TEXT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.8,
            ),
        )
    return st.session_state.chat


# =====================================================================================
# 4. IMAGE GENERATION / DESIGN HELPER
# =====================================================================================
def generate_image(client: genai.Client, prompt: str, reference_image: Image.Image | None = None):
    """
    Calls Google's Imagen model via google-genai SDK.
    Returns (pil_image_or_None, caption_text).
    """
    # Use generate_images for Imagen models
    response = client.models.generate_images(
        model=IMAGE_GEN_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
        )
    )

    pil_image = None
    if response.generated_images:
        # Convert generated bytes to PIL Image
        pil_image = Image.open(io.BytesIO(response.generated_images[0].image.image_bytes))

    return pil_image, f"✨ Here is your generated image for: '{prompt}'"
def looks_like_image_request(text: str) -> bool:
    return bool(IMAGE_INTENT_PATTERN.search(text or ""))
# =====================================================================================
# 5. STREAMLIT PAGE CONFIG & STYLE
# =====================================================================================
st.set_page_config(page_title="Aura — AI Assistant", page_icon="✨", layout="centered")

st.markdown(
    """
    <style>
      .block-container {max-width: 820px;}
      [data-testid="stChatMessage"] {border-radius: 14px; padding: 0.4rem 0.2rem;}
      h1 {margin-bottom: 0.1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✨ Aura")
st.caption("Your advanced, personalized AI assistant — text, vision & image design in one chat.")


# =====================================================================================
# 6. SIDEBAR — SETTINGS, CAMERA INPUT, OPTIONS
# =====================================================================================
api_key = get_api_key()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📷 Camera Input")
    camera_image = st.camera_input("Capture a photo to discuss or edit")

    st.markdown("---")
    st.markdown("### 🎨 Image Design Mode")
    force_image_mode = st.toggle(
        "Force image generation for next message",
        help="Turn this on if you want Aura to treat your next message purely as an "
             "image-generation/design request, regardless of phrasing.",
    )

    st.markdown("---")
    if st.button("🗑️ Clear conversation"):
        for k in ("messages", "chat"):
            st.session_state.pop(k, None)
        st.rerun()

if not api_key:
    st.info("👋 Add your Gemini API key in the sidebar to start chatting with Aura.")
    st.stop()

client = get_client(api_key)
chat = get_chat_session(client)

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role", "text", "image": PIL.Image|None}


# =====================================================================================
# 7. RENDER CHAT HISTORY
# =====================================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image") is not None:
            st.image(msg["image"], use_container_width=True)
        if msg.get("text"):
            st.markdown(msg["text"])


# =====================================================================================
# 8. CHAT INPUT & RESPONSE HANDLING
# =====================================================================================
user_text = st.chat_input("Message Aura... (e.g. 'design a logo for a coffee shop')")

if user_text:
    captured_pil = Image.open(camera_image) if camera_image is not None else None

    # --- Display user's turn immediately ---
    st.session_state.messages.append(
        {"role": "user", "text": user_text, "image": captured_pil}
    )
    with st.chat_message("user"):
        if captured_pil is not None:
            st.image(captured_pil, use_container_width=True)
        st.markdown(user_text)

    want_image = force_image_mode or looks_like_image_request(user_text)

    # --- Assistant's turn ---
    with st.chat_message("assistant"):
        if want_image:
            with st.spinner("Designing your image..."):
                try:
                    image_out, caption = generate_image(client, user_text, captured_pil)
                except Exception as e:
                    image_out, caption = None, f"⚠️ Image generation failed: {e}"

            if image_out is not None:
                st.image(image_out, use_container_width=True)
            if caption:
                st.markdown(caption)
            if image_out is None and not caption:
                caption = "⚠️ I couldn't produce an image for that request. Try rephrasing it."
                st.markdown(caption)

            st.session_state.messages.append(
                {"role": "assistant", "text": caption, "image": image_out}
            )

        else:
            # Build multimodal message: text (+ camera image if provided)
            message_parts = [user_text] if captured_pil is None else [captured_pil, user_text]

            placeholder = st.empty()
            full_reply = ""
            try:
                stream = chat.send_message_stream(message=message_parts)
                for chunk in stream:
                    if chunk.text:
                        full_reply += chunk.text
                        placeholder.markdown(full_reply + "▌")
                placeholder.markdown(full_reply)
            except Exception as e:
                full_reply = f"⚠️ Something went wrong: {e}"
                placeholder.markdown(full_reply)

            st.session_state.messages.append(
                {"role": "assistant", "text": full_reply, "image": None}
            )
