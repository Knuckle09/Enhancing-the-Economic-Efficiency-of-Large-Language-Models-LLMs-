# Run Nimbus AI

Nimbus AI uses one lightweight API for Gemini and Ollama. The hosted service uses Gemini; a local API can reach Ollama models installed on your computer.

## Run locally with Ollama and Gemini

Requirements: Python 3.10–3.12, Node.js 20 or newer, and Ollama if you want local models.

1. Install Ollama, start its service, then pull the models you want to use. On Windows, the Ollama app normally starts the service for you.

   ```powershell
   ollama pull qwen2.5-coder:3b
   ollama pull qwen2-math:latest
   ollama pull phi3:latest
   ollama pull tinyllama:latest
   ollama pull nomic-embed-text
   ollama list
   ```

   `nomic-embed-text` is optional and enables the report-aligned semantic similarity metric for local Ollama requests. Ollama's embedding API detects it automatically; another embedding model can be selected with `OLLAMA_EMBEDDING_MODEL`.

2. Open PowerShell at the repository root and install the API dependencies:

   ```powershell
   cd Backend/production
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements-cloud.txt
   ```

3. In that same terminal, configure Gemini if you want cloud models. This is optional when you only want Ollama.

   ```powershell
   $env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   $env:OLLAMA_ENABLED = "true"
   ```

4. Start the backend:

   ```powershell
   python api.py
   ```

   The API runs at `http://127.0.0.1:5000`. Keep this terminal open. It detects installed Ollama models and advertises them to the frontend.

5. In a second PowerShell window, configure and start the frontend:

   ```powershell
   cd Frontend/nimbus-ai/nimbus-ai
   npm ci
   Copy-Item .env.local.example .env.local
   npm run dev
   ```

   Open the Vite URL printed in that window, normally `http://localhost:5173`. Select **Auto** for adaptive routing or **Manual** to choose an installed Gemini or Ollama model. The strategy selector controls safe prompt cleanup.

## Report features in the running app

- Prompt category detection, conservative/balanced/aggressive cleanup, and estimated token/cost comparison.
- Automatic model routing, manual overrides, local Ollama, Gemini, and quota fallback from Pro to Flash.
- Recent conversation turns sent to the model and up to 60 chat messages saved in this browser's local storage.
- Live model output streams into the chat as it is generated; the request stays open for long-form answers and sends periodic keepalives.
- A bounded near-duplicate response cache and thumbs-up/down feedback that adjusts automatic model routing while the API process is running.
- Analytics from saved replies: token reduction, embedding-based semantic similarity, estimated cost savings, token totals, model utilization, response time, strategy results, and prompt details.
- A separate report benchmark panel shows the study's 30% token reduction, 25-35% cost savings, similarity above 0.90 (0.92 in the comparison table), and 22% multi-turn coherence improvement. These are reported study results, not live session readings.

The lightweight optimizer avoids rewriting code and uses meaning-preserving cleanup rules. Gemini token counts use the provider tokenizer; Ollama prompt counts use an approximate four-characters-per-token estimate. Gemini requests use Gemini text embeddings for semantic similarity, while local Ollama requests use an installed Ollama embedding model; the score is shown as not measured if the embedding service is unavailable or a prompt exceeds the live embedding limit. Gemini cost and embedding cost are estimates at standard paid-tier list prices; free-tier credits and actual billing can differ. Ollama inference has no API list-price estimate. The response cache and routing feedback are held in API memory and reset when the service restarts. Chat content stays in this browser and is sent to the selected model provider when you submit a prompt.

## Hosted version

- Public frontend: [GitHub Pages](https://knuckle09.github.io/Enhancing-the-Economic-Efficiency-of-Large-Language-Models-LLMs-/)
- Gemini API: `https://llm-efficiency-backend.onrender.com`
- Frontend changes pushed to `main` deploy through `.github/workflows/deploy-pages.yml`.
- Render runs `cloud_api.py` with `requirements-cloud.txt`. Set `GEMINI_API_KEY` in the Render service environment. The hosted instance has Ollama disabled because it cannot reach models on visitors' computers.
- Render's free service sleeps after 15 minutes without traffic and can take about a minute to wake. The chat shows wake-up progress and streams the answer once generation begins. The Render start command allows up to 10 minutes for long replies. Upgrade the Render instance to avoid cold starts for consistently fast availability.

To use Ollama, run the local API and frontend using the steps above. A public web page cannot directly access Ollama installed on your computer.

Never commit `.env`, `.env.local`, or API keys.
