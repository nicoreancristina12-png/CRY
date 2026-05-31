"""
ArturCecanAI — Telegram assistant for Școala de șoferi Artur Cecan, Iași.

Stack:
  - python-telegram-bot (async) for Telegram integration
  - OpenAI API for replies
  - Zep Cloud for long-term conversation memory per user

Each Telegram chat_id maps 1:1 to a Zep user + session, so the bot remembers
the conversation across restarts and across days.
"""

import logging
import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from zep_cloud.client import AsyncZep
from zep_cloud.types import Message as ZepMessage

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ZEP_API_KEY = os.environ["ZEP_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
# python-telegram-bot is chatty at INFO; quiet it down a notch
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ArturCecanAI")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
zep_client = AsyncZep(api_key=ZEP_API_KEY)

# ---------------------------------------------------------------------------
# System prompt — the assistant's "personality" and knowledge about the school
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ești ArturCecanAI, asistentul virtual al Școlii de șoferi Artur Cecan din Iași.

DESPRE ȘCOALĂ:
- Locație: Bd. Dacia nr. 45, Iași
- Contact: WhatsApp 0772 222 345 / 0770 881 145
- Website: arturcecan.ro
- Specialitate: educație rutieră completă — nu doar permis, ci conducere defensivă,
  reacție în condiții extreme, prim ajutor și mindsetul corect pentru drum.
- Diferențiator: aici nu se țipă la cursanți. Răbdare și pregătire reală.

CUM RĂSPUNZI:
- Cald, scurt, în limba română. Mesaje pentru Telegram — scurte, fără paragrafe lungi.
- Răspunzi la întrebări despre: înscriere, program, ce categorii sunt disponibile,
  diferențierea școlii, trasee de examen din Iași, frica de a conduce, mituri.
- NU inventa prețuri, date exacte ale seriilor sau promoții. Dacă te întreabă ceva
  ce nu știi sigur, spui: „Pentru asta te rog sună sau scrie pe WhatsApp 0772 222 345 —
  îți răspunde Cristina, secretara noastră.”
- La sfârșit, când e cazul, inviți discret la WhatsApp sau să se înscrie.
- Dacă cineva pare interesat să se înscrie, ceri politicos numele și un număr de
  telefon, ca să-l contacteze cineva din echipă.
- Folosești emoji-uri cu măsură (1-2 per mesaj max).

LIMITE:
- Nu dai sfaturi medicale, legale sau de altă natură nelegată de școală.
- Dacă cineva e agresiv sau insistă pe ceva nepotrivit, rămâi politicos și redirecționezi
  spre WhatsApp pentru discuții directe."""

MAX_HISTORY_MESSAGES = 20  # how many recent messages we pull from Zep for context


# ---------------------------------------------------------------------------
# Zep helpers
# ---------------------------------------------------------------------------

def _user_id(chat_id: int) -> str:
    """Stable Zep user id from Telegram chat id."""
    return f"tg_{chat_id}"


def _session_id(chat_id: int) -> str:
    """One session per chat — keeps the whole history together."""
    return f"tg_session_{chat_id}"


async def ensure_zep_user_and_session(chat_id: int, telegram_user) -> None:
    """Create the Zep user + session on first contact. Idempotent."""
    user_id = _user_id(chat_id)
    session_id = _session_id(chat_id)

    # User
    try:
        await zep_client.user.add(
            user_id=user_id,
            first_name=getattr(telegram_user, "first_name", None) or "",
            last_name=getattr(telegram_user, "last_name", None) or "",
            email=None,
            metadata={
                "telegram_username": getattr(telegram_user, "username", None) or "",
                "source": "telegram",
            },
        )
        logger.info("Created Zep user %s", user_id)
    except Exception as e:
        # Already exists → fine
        logger.debug("Zep user add skipped for %s: %s", user_id, e)

    # Session
    try:
        await zep_client.memory.add_session(
            session_id=session_id,
            user_id=user_id,
        )
        logger.info("Created Zep session %s", session_id)
    except Exception as e:
        logger.debug("Zep session add skipped for %s: %s", session_id, e)


async def load_history(chat_id: int) -> List[Dict[str, str]]:
    """Pull recent conversation from Zep and format for OpenAI."""
    session_id = _session_id(chat_id)
    try:
        memory = await zep_client.memory.get(session_id=session_id)
    except Exception as e:
        logger.warning("Could not load Zep memory for %s: %s", session_id, e)
        return []

    messages: List[Dict[str, str]] = []
    for m in (memory.messages or [])[-MAX_HISTORY_MESSAGES:]:
        role = "assistant" if m.role_type == "assistant" else "user"
        messages.append({"role": role, "content": m.content})
    return messages


async def save_turn(chat_id: int, user_text: str, assistant_text: str) -> None:
    """Persist the latest user/assistant exchange to Zep."""
    session_id = _session_id(chat_id)
    try:
        await zep_client.memory.add(
            session_id=session_id,
            messages=[
                ZepMessage(role="user", role_type="user", content=user_text),
                ZepMessage(role="assistant", role_type="assistant", content=assistant_text),
            ],
        )
    except Exception as e:
        logger.error("Failed to save turn to Zep for %s: %s", session_id, e)


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

async def generate_reply(history: List[Dict[str, str]], user_text: str) -> str:
    """Call OpenAI with system prompt + history + new user message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    response = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.6,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

WELCOME = (
    "Salut! 👋 Sunt asistentul virtual al Școlii de șoferi *Artur Cecan* din Iași.\n\n"
    "Te pot ajuta cu informații despre înscriere, serii, ce înveți la noi și "
    "diferențele față de alte școli.\n\n"
    "Întreabă-mă orice — sau scrie /reset dacă vrei să o luăm de la capăt."
)


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await ensure_zep_user_and_session(chat_id, update.effective_user)
    await update.message.reply_text(WELCOME, parse_mode="Markdown")


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Comenzi disponibile:\n"
        "/start — pornire\n"
        "/reset — uită conversația și o ia de la capăt\n"
        "/contact — date de contact ale școlii\n\n"
        "Sau scrie-mi pur și simplu ce vrei să afli."
    )


async def cmd_contact(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📍 Bd. Dacia nr. 45, Iași\n"
        "📞 WhatsApp: 0772 222 345 / 0770 881 145\n"
        "🌐 arturcecan.ro"
    )


async def cmd_reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session_id = _session_id(chat_id)
    try:
        await zep_client.memory.delete(session_id=session_id)
    except Exception as e:
        logger.warning("Reset failed for %s: %s", session_id, e)
    await ensure_zep_user_and_session(chat_id, update.effective_user)
    await update.message.reply_text("Gata, am uitat conversația. Cu ce te ajut? 🙂")


async def on_message(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    await ensure_zep_user_and_session(chat_id, update.effective_user)
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        history = await load_history(chat_id)
        reply = await generate_reply(history, user_text)
        await save_turn(chat_id, user_text, reply)
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Failed to handle message from chat %s", chat_id)
        await update.message.reply_text(
            "Îmi pare rău, am avut o problemă tehnică. Te rog încearcă din nou "
            "sau scrie pe WhatsApp 0772 222 345."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("contact", cmd_contact))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("ArturCecanAI is starting (model=%s)...", OPENAI_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
