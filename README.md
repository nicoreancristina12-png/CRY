# ArturCecanAI

Telegram assistant for **Școala de șoferi Artur Cecan, Iași**. Answers prospective students 24/7 about enrollment, schedules, what makes the school different, and routes serious inquiries to the school's WhatsApp.

Stack: Python · python-telegram-bot · OpenAI · Zep Cloud (long-term memory).

## What it does

- Answers questions in Romanian, in the school's voice (warm, short, no shouting).
- Remembers each user's conversation **across days and restarts** via Zep — so a returning lead doesn't have to repeat themselves.
- Never invents prices or series dates — pushes the user to WhatsApp `0772 222 345` when uncertain.
- Per-chat reset with `/reset`.

## Setup

### 1. Get the keys

- **Telegram bot token** → talk to [@BotFather](https://t.me/BotFather) on Telegram, run `/newbot`, copy the token.
- **OpenAI API key** → https://platform.openai.com/api-keys
- **Zep Cloud API key** → https://app.getzep.com (free tier is fine to start)

### 2. Install

```bash
git clone <this-repo>
cd artur_cecan_ai
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and paste your keys
```

### 4. Run

```bash
python main.py
```

You'll see `ArturCecanAI is starting...` in the logs. Open Telegram, find your bot, send `/start`.

## Commands

| Command | Effect |
|---------|--------|
| `/start` | Greeting + creates the user in Zep |
| `/help` | Lists available commands |
| `/contact` | Drops the school's address, WhatsApp, website |
| `/reset` | Wipes the chat history from Zep and starts fresh |

Anything else is treated as a normal chat message and goes through OpenAI with the Zep-backed conversation history.

## How memory works

- Each Telegram `chat_id` maps to a Zep user (`tg_<chat_id>`) and a single session (`tg_session_<chat_id>`).
- On every message: load the last 20 messages from Zep → prepend the system prompt → call OpenAI → save the new turn back to Zep.
- The system prompt (school facts, tone, guardrails) lives in `main.py` as `SYSTEM_PROMPT`. Edit it there to change the assistant's behavior.

## Tuning

- **Model**: set `OPENAI_MODEL` in `.env`. Default is `gpt-4o-mini` (cheap, fast). For higher quality use `gpt-4o`.
- **History length**: change `MAX_HISTORY_MESSAGES` in `main.py`.
- **Temperature / max_tokens**: adjust in `generate_reply()`.

## Deploying

The bot uses long polling, so any machine that can stay online works:

- A small VPS (DigitalOcean, Hetzner): run with `nohup python main.py &` or under `systemd`.
- A free tier on Railway / Render / Fly.io — connect the repo, set the env vars in the dashboard, deploy.

## Project structure

```
artur_cecan_ai/
├── main.py            # the bot
├── requirements.txt
├── .env.example
├── .env               # your real keys — DO NOT commit
└── README.md
```

## Roadmap ideas

- Hand off to a human agent when the user wants to enroll (forward to a Telegram group with school staff).
- Auto-collect leads (name + phone) into a Google Sheet or CRM.
- Voice messages: transcribe with Whisper, reply with TTS.
- Send media (photos of the office, location pin) when relevant.
