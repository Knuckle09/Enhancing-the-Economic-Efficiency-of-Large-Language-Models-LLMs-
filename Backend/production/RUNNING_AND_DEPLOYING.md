# Run Nimbus AI

## Run the full original backend locally

Use Python 3.10–3.12 and Node.js 20 or newer. The full backend uses the original RL/NLP modules. Its first start can download NLP model files.

1. Install [Ollama](https://ollama.com/download) if you want local models, then pull the models you plan to use:

   ```powershell
   ollama pull codellama:7b
   ollama pull qwen2-math:latest
   ollama pull tinyllama:latest
   ollama pull phi3:latest
   ```

2. In PowerShell, from the repository root, create and activate a Python environment and install the backend dependencies:

   ```powershell
   cd Backend/production
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Set your Gemini key in the same terminal if you want Gemini cloud models:

   ```powershell
   $env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
   ```

4. Start Ollama in another terminal (`ollama serve` if it is not already running), then start the original backend:

   ```powershell
   cd Backend/production
   .\.venv\Scripts\Activate.ps1
   python api.py
   ```

   The local API is at `http://localhost:5000`. Gemini is available when `GEMINI_API_KEY` is set; Ollama is available when its server and the selected models are installed on this computer.

5. In another terminal, start the frontend and point it at the local backend:

   ```powershell
   cd Frontend/nimbus-ai/nimbus-ai
   npm ci
   $env:VITE_API_URL = "http://localhost:5000"
   $env:VITE_ENABLE_OLLAMA_MODELS = "true"
   npm run dev
   ```

   Open the local URL printed by Vite (normally `http://localhost:5173`). Choose **Manual** to select Gemini or an Ollama model. In **Auto**, the original router prefers Gemini when configured and otherwise routes to Ollama.

## Deploy the public frontend and Gemini API

- The frontend is built and published to GitHub Pages by `.github/workflows/deploy-pages.yml` whenever frontend files change on `main`.
- Render runs `cloud_api.py` with `requirements-cloud.txt` to fit the free instance. In the Render service environment, set `GEMINI_API_KEY` to your own key. The hosted service supports Gemini; Ollama is only available when the full original API runs on a computer that has Ollama installed.
- The hosted API URL is `https://llm-efficiency-backend.onrender.com`. Render's free instance can sleep while idle, so the first request may take a short time to wake it.

Never commit `.env` files or API keys.
