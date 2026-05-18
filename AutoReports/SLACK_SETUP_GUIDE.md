# Chorus Slack Bot Setup Guide

This guide walks you through integrating Chorus with Slack so your team can interact with the Knowledge Lab knowledge base directly from Slack.

## Overview

The Chorus Slack bot:
- Responds to @mentions in any channel it's invited to
- Searches the RAG knowledge base to answer questions
- Uses Claude to generate intelligent, context-aware responses
- Runs via Slack's Socket Mode (no public URL required)

---

## Prerequisites

Before starting, ensure you have:
- [ ] A Slack workspace where you have permission to install apps
- [ ] An Anthropic API key (`ANTHROPIC_API_KEY`)
- [ ] The Chorus RAG server running (or ability to run it)
- [ ] Python 3.10+ with required packages

---

## Step 1: Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App**
3. Choose **From scratch**
4. Enter:
   - **App Name**: `Chorus` (or your preferred name)
   - **Workspace**: Select your workspace
5. Click **Create App**

---

## Step 2: Enable Socket Mode

Socket Mode allows the bot to receive events without a public URL.

1. In your app settings, go to **Socket Mode** (left sidebar)
2. Toggle **Enable Socket Mode** to ON
3. When prompted, create an app-level token:
   - **Token Name**: `chorus-socket`
   - **Scopes**: `connections:write` (should be pre-selected)
4. Click **Generate**
5. **Save this token** - it starts with `xapp-` and is your `SLACK_APP_TOKEN`

---

## Step 3: Configure Bot Permissions

1. Go to **OAuth & Permissions** (left sidebar)
2. Scroll to **Scopes** → **Bot Token Scopes**
3. Add these scopes:
   - `app_mentions:read` - Allows bot to see when it's mentioned
   - `chat:write` - Allows bot to send messages
   - `channels:history` - (Optional) Read channel messages for context
   - `groups:history` - (Optional) Read private channel messages
   - `im:history` - (Optional) Read DM history
   - `im:write` - (Optional) Send DMs

---

## Step 4: Subscribe to Events

1. Go to **Event Subscriptions** (left sidebar)
2. Toggle **Enable Events** to ON
3. Expand **Subscribe to bot events**
4. Add these events:
   - `app_mention` - Triggers when someone @mentions your bot
   - `message.im` - (Optional) Triggers on direct messages
5. Click **Save Changes**

---

## Step 5: Install App to Workspace

1. Go to **Install App** (left sidebar)
2. Click **Install to Workspace**
3. Review permissions and click **Allow**
4. **Save the Bot User OAuth Token** - it starts with `xoxb-` and is your `SLACK_BOT_TOKEN`

---

## Step 6: Set Environment Variables

Create a `.env` file in your chorus directory or export these variables:

```bash
# Required
export SLACK_BOT_TOKEN="xoxb-your-bot-token-here"
export SLACK_APP_TOKEN="xapp-your-app-token-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Optional (defaults shown)
export RAG_SERVER_URL="http://localhost:8765"
```

### Finding Your Tokens

| Token | Where to Find It | Starts With |
|-------|------------------|-------------|
| `SLACK_BOT_TOKEN` | OAuth & Permissions → Bot User OAuth Token | `xoxb-` |
| `SLACK_APP_TOKEN` | Basic Information → App-Level Tokens | `xapp-` |

---

## Step 7: Install Python Dependencies

The Slack bot requires additional packages not in the base requirements:

```bash
pip install slack-bolt aiohttp anthropic
```

Or add to requirements.txt:
```
slack-bolt>=1.18.0
aiohttp>=3.9.0
anthropic>=0.40.0
```

---

## Step 8: Start the Services

### Option A: Run Locally (Development)

1. **Start the RAG server** (in one terminal):
   ```bash
   python rag_http_server.py
   # Or the async version:
   python rag_http_server_async.py
   ```

2. **Start the Slack bot** (in another terminal):
   ```bash
   # Make sure environment variables are set
   python slack_bot.py
   ```

   You should see:
   ```
   Connected to RAG server (2000 chunks indexed)
   CHORUS Slack bot starting...
   ```

### Option B: Run with Docker

```bash
# Start both RAG server and Slack bot
docker-compose --profile rag --profile slack up -d

# View logs
docker-compose logs -f slack-bot
```

Make sure your `.env` file is in the same directory as `docker-compose.yml`, or pass variables directly:

```bash
SLACK_BOT_TOKEN=xoxb-... SLACK_APP_TOKEN=xapp-... ANTHROPIC_API_KEY=sk-ant-... \
  docker-compose --profile rag --profile slack up
```

---

## Step 9: Invite the Bot to Channels

1. Go to any Slack channel where you want Chorus available
2. Type `/invite @Chorus` (or whatever you named your bot)
3. The bot is now listening in that channel

---

## Step 10: Test the Bot

In a channel where the bot is present:

```
@Chorus What research projects are currently active?
```

```
@Chorus Who works on computational social science?
```

```
@Chorus Tell me about recent grants
```

The bot should respond in a thread with information from the knowledge base.

---

## Troubleshooting

### Bot doesn't respond

1. **Check the bot is running**:
   ```bash
   # If running locally, check the terminal for errors
   # If Docker:
   docker-compose logs slack-bot
   ```

2. **Verify environment variables are set**:
   ```bash
   echo $SLACK_BOT_TOKEN  # Should start with xoxb-
   echo $SLACK_APP_TOKEN  # Should start with xapp-
   ```

3. **Check the bot is in the channel**: Type `/invite @Chorus`

4. **Verify Socket Mode is enabled** in Slack app settings

### "RAG server not reachable" warning

The bot will still start, but queries will fail. Ensure:
- The RAG server is running on port 8765
- If using Docker, both services are on the same network
- Check: `curl http://localhost:8765/health`

### "Missing environment variables" error

Set all three required variables:
```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Bot responds with errors

Check the Anthropic API key is valid:
```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

---

## Architecture

```
Slack Workspace
      │
      │ (Socket Mode - WebSocket)
      ▼
┌─────────────────┐
│   slack_bot.py  │
│  (Slack Bolt)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Claude API     │◄────│  RAG Server     │
│  (Anthropic)    │     │  (port 8765)    │
└─────────────────┘     └─────────────────┘
         │                      │
         │                      ▼
         │              ┌─────────────────┐
         │              │  FAISS + BM25   │
         │              │  Knowledge Base │
         └──────────────┴─────────────────┘
```

---

## What the Bot Can Do

| Query Type | Example |
|------------|---------|
| Research projects | "What are the current research projects?" |
| People | "Who is working on AI ethics?" |
| Grants | "Tell me about NSF grants" |
| Workshops | "When was the last workshop?" |
| Publications | "What papers have been published recently?" |
| General questions | "What does Knowledge Lab do?" |

---

## Security Notes

- **Never commit tokens to git** - Use environment variables or `.env` files
- Add `.env` to your `.gitignore`
- Rotate tokens if they're ever exposed
- The bot only responds to @mentions, not all messages
- Consider restricting which channels the bot can access

---

## Next Steps

Once the bot is running, you might want to:

1. **Customize the system prompt** in `slack_bot.py` (line 125-136)
2. **Add slash commands** for specific actions
3. **Enable DM support** for private conversations
4. **Add logging** for debugging and analytics
5. **Set up monitoring** to ensure the bot stays online

---

## Quick Reference

| Component | Command/Location |
|-----------|-----------------|
| Start RAG server | `python rag_http_server.py` |
| Start Slack bot | `python slack_bot.py` |
| Docker (both) | `docker-compose --profile rag --profile slack up` |
| Slack App Settings | https://api.slack.com/apps |
| Test RAG server | `curl http://localhost:8765/health` |
