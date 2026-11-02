# Nimbus AI

**Live app:** [GitHub Pages](https://knuckle09.github.io/Enhancing-the-Economic-Efficiency-of-Large-Language-Models-LLMs-/)
**Run locally with Gemini and Ollama:** [step-by-step setup](Backend/production/RUNNING_AND_DEPLOYING.md)

Nimbus AI is a chat app based on the optimization and routing goals described in the project report. It connects to Gemini in the hosted deployment and to Gemini or locally installed Ollama models when run on a computer with Ollama.

## Current features

- Auto and manual model selection, with task-category and complexity-aware routing.
- Gemini Flash and Pro, Ollama model discovery, and Gemini Pro quota fallback to Flash.
- Safe prompt cleanup, duplicate-sentence pruning, selectable optimization strategy, and approximate token/cost analytics.
- Recent-turn conversation context sent with each request; chat history is saved in the browser's local storage.
- A bounded cache for highly similar standalone prompts and thumbs-up/down feedback that updates the in-process routing policy.
- Analytics from real saved replies: token usage, text-overlap proxy, paid-tier Gemini cost estimates, strategy results, response timing by model, and prompt details.

The serving API is intentionally lightweight; it does not load the research/training stack at startup. Token counts are rough estimates and the text-overlap score is not an embedding-based semantic score. The paper's evaluation results are reported research results; the live dashboard shows measurements from your own saved replies. Cache entries and feedback learning reset when the API process restarts.

## Run locally

Follow [Backend/production/RUNNING_AND_DEPLOYING.md](Backend/production/RUNNING_AND_DEPLOYING.md). In short: install Ollama and the models you want, start `Backend/production/api.py`, configure the frontend's `.env.local` to use `http://127.0.0.1:5000`, then start the Vite frontend. The public GitHub Pages site cannot reach Ollama installed on a visitor's computer.

## Hosted deployment

- The frontend is published to GitHub Pages by `.github/workflows/deploy-pages.yml` on pushes to `main`.
- The Gemini backend runs on Render from `Backend/production/cloud_api.py` using `requirements-cloud.txt`.
- Set `GEMINI_API_KEY` in the Render environment. Ollama is disabled on Render; the deployed service cannot reach local machines.
- Render's free instance can sleep while idle, so the first API request may take longer while it wakes.

Never commit API keys or `.env` files.
