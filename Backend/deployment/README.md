# RL Text Optimization Framework - Deployment Package

This deployment package contains everything you need to run the RL-based text optimization API server with Gemini integration.

## 📦 What's Included

- **Core Files:**
  - `api.py` - Main API server
  - `llm_efficiency_test.py` - Optimization engine
  - `prompt_diversity_test.py` - LLM routing
  - `rl_optimizer.py` - RL training
  - `txtprprc1.py` - Text preprocessing
  - `run.py` - Main framework entry
  - `frontend_demo.html` - Web interface

- **Configuration:**
  - `.env` - API keys (GEMINI_API_KEY)
  - `requirements.txt` - Python dependencies

- **Data & Models:**
  - `models/` - Pre-trained models
  - `prompts/` - Training prompts
  - `results/` - Training data

- **Scripts:**
  - `setup.bat` / `setup.sh` - One-time setup
  - `start_server.bat` / `start_server.sh` - Start API server

## 🚀 Quick Start

### Windows

1. **First Time Setup:**
   ```cmd
   setup.bat
   ```
   
2. **Edit .env file with your API key:**
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Start the Server:**
   ```cmd
   start_server.bat
   ```

4. **Open the Frontend:**
   - Open `frontend_demo.html` in your browser
   - Or visit `http://localhost:5000`

### Linux/Mac

1. **First Time Setup:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Edit .env file with your API key:**
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Start the Server:**
   ```bash
   ./start_server.sh
   ```

4. **Open the Frontend:**
   - Open `frontend_demo.html` in your browser
   - Or visit `http://localhost:5000`

## 🔑 Getting Your Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Create or select a project
3. Click "Create API Key"
4. Copy the key and paste it in `.env` file

## 📊 Features

- **Automatic Prompt Optimization:** Reduces token count by 30-70% while preserving meaning
- **Smart LLM Routing:** Automatically selects best model for your prompt
- **Multi-Model Support:**
  - **Cloud:** Gemini 2.5 Pro, Gemini 2.5 Flash
  - **Local:** CodeLlama, Qwen-Math, TinyLlama, Phi-3
- **Cost Tracking:** Shows cost savings with Gemini pricing
- **Real-time Metrics:** Token reduction, similarity, processing time

## 🛠️ Manual Setup (Alternative)

If scripts don't work, follow these steps:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data:**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

3. **Download spaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Create .env file:**
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Start Server:**
   ```bash
   python api.py
   ```

## 🔧 Configuration

### Environment Variables (.env)

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### API Endpoints

- `GET /api/health` - Check server status
- `POST /api/process` - Optimize prompt
- `GET /api/strategies` - List available strategies

### Example API Request

```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "model_preference": "auto",
    "include_response": true
  }'
```

## 📝 Requirements

- Python 3.8+
- 4GB+ RAM
- Internet connection (for Gemini API)
- Ollama installed (optional, for local models)

## 🐛 Troubleshooting

**Server won't start:**
- Check if `.env` file exists and has valid API key
- Verify Python dependencies are installed: `pip list`
- Check port 5000 is not in use

**NLTK errors:**
- Run: `python -c "import nltk; nltk.download('punkt_tab')"`

**Gemini API errors:**
- Verify API key is correct
- Check internet connection
- Confirm API quota: https://console.cloud.google.com

**Frontend not loading:**
- Make sure server is running on port 5000
- Check browser console for errors
- Try opening `frontend_demo.html` directly

## 📚 Documentation

For detailed documentation, see:
- API Guide: Check `api.py` docstrings
- Model Selection: See `prompt_diversity_test.py`
- Optimization Strategies: See `llm_efficiency_test.py`

## 💡 Tips

- Use **Framework Suggestion** mode for best results
- **Gemini Pro** for complex tasks (coding, math)
- **Gemini Flash** for simple tasks (faster, cheaper)
- Enable **Generate LLM response** to see actual output
- Monitor **Cost Saved** to track savings

## 📊 Performance

- Average optimization: 30-70% token reduction
- Semantic similarity: 85-95%
- Processing time: 1-5 seconds
- Cost savings: Up to 70% on API calls

## 🔒 Security

- Never commit `.env` file to version control
- Keep API keys secure
- Use environment variables in production
- Rotate API keys periodically

## 📞 Support

For issues or questions:
- Check logs in terminal/console
- Review error messages carefully
- Ensure all dependencies are installed
- Verify API key permissions

---

**Ready to optimize your prompts!** 🚀
