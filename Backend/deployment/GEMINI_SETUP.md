# Gemini API Integration Guide

This guide will help you set up and use Google's Gemini AI models in your RL-based text optimization framework.

## Table of Contents
- [Getting Your API Key](#getting-your-api-key)
- [Setting Up Gemini](#setting-up-gemini)
- [Usage Examples](#usage-examples)
- [Available Models](#available-models)
- [Troubleshooting](#troubleshooting)

---

## Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key" or "Create API Key"
4. Copy your API key (keep it secure!)

**Important**: Never commit your API key to version control!

---

## Setting Up Gemini

### Method 1: Using the Setup Script (Recommended)

```bash
# Install the required package
pip install google-generativeai

# Set up your API key and save to .env file
python setup_gemini.py --api-key YOUR_API_KEY_HERE --save-env

# Test the connection
python setup_gemini.py --test
```

### Method 2: Manual Setup

#### On Linux/Mac:
```bash
# Install package
pip install google-generativeai

# Set environment variable (temporary - current session only)
export GEMINI_API_KEY='your-api-key-here'

# Make it permanent (add to ~/.bashrc or ~/.zshrc)
echo "export GEMINI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

#### On Windows PowerShell:
```powershell
# Install package
pip install google-generativeai

# Set environment variable (temporary)
$env:GEMINI_API_KEY='your-api-key-here'

# Make it permanent
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-api-key-here", "User")
```

#### On Windows Command Prompt:
```cmd
pip install google-generativeai
setx GEMINI_API_KEY "your-api-key-here"
```

### Method 3: Using .env File

1. Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your-api-key-here
```

2. Install python-dotenv:
```bash
pip install python-dotenv
```

3. Load in your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage Examples

### Basic Text Generation

```python
from llm_efficiency_test import LLMEfficiencyTest

# Initialize
tester = LLMEfficiencyTest()

# Generate with Gemini
response = tester.generate_response(
    "Explain quantum computing in simple terms",
    llm="gemini-pro",
    max_tokens=256
)
print(response)
```

### Prompt Optimization with Gemini

```python
from llm_efficiency_test import LLMEfficiencyTest

tester = LLMEfficiencyTest()

original_prompt = """
Please write a comprehensive Python function that can calculate 
the factorial of any given positive integer number. The function 
should handle edge cases properly.
"""

# Optimize using Gemini
optimized_prompt, metrics = tester.optimize_tokens(
    original_prompt,
    target_reduction=0.30,
    min_similarity=0.85,
    llm="gemini-pro",
    category="coding"
)

print(f"Original: {original_prompt}")
print(f"Optimized: {optimized_prompt}")
print(f"Reduction: {metrics['reduction_percent']:.1f}%")
print(f"Similarity: {metrics['similarity']:.3f}")
```

### Smart Routing with Gemini Preference

```python
from prompt_diversity_test import PromptDiversityTester

tester = PromptDiversityTester()

# Automatically route to best LLM (prefers Gemini for coding/math/generic)
prompt = "Write a binary search algorithm"
llm = tester.route_prompt_to_llm(prompt, prefer_gemini=True)
print(f"Routing to: {llm}")  # Will use gemini-pro for coding tasks

# Process with automatic routing
result = tester.process_prompt(prompt)
```

### Using Gemini in API Server

```bash
# Start the API server
python api.py

# The API will automatically detect Gemini availability
# and include it in the /api/strategies endpoint
```

Test with curl:
```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function for bubble sort",
    "include_response": true
  }'
```

### Comparing Models

```python
from llm_efficiency_test import LLMEfficiencyTest

tester = LLMEfficiencyTest()
prompt = "Explain recursion"

# Test with different models
for llm in ["tinyllama", "codellama", "gemini-pro"]:
    response = tester.generate_response(prompt, llm=llm)
    print(f"\n{llm}:")
    print(response[:200])
```

---

## Available Models

### In This Framework

| Model ID | Full Name | Best For | Speed | Quality |
|----------|-----------|----------|-------|---------|
| `gemini-pro` | Gemini Pro | Coding, Math, Reasoning | Medium | High |
| `gemini-flash` | Gemini 1.5 Flash | Quick tasks, General use | Fast | Good |

### Aliases

- `gemini` → `gemini-pro`
- `gemini-pro` → Full Gemini Pro model
- `gemini-flash` → Gemini 1.5 Flash model

### Usage in Code

```python
# All these work:
tester.generate_response(prompt, llm="gemini")
tester.generate_response(prompt, llm="gemini-pro")
tester.generate_response(prompt, llm="gemini-flash")
```

---

## Running Tests

### Quick Test
```bash
python test_gemini.py
```

This will run:
1. Basic generation test
2. Prompt optimization test
3. Smart routing test
4. Model comparison test

### Manual Testing

```bash
# Test connection only
python setup_gemini.py --test

# Show available models
python setup_gemini.py --models
```

---

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution:**
```bash
# Check if it's set
echo $GEMINI_API_KEY  # Linux/Mac
echo $env:GEMINI_API_KEY  # Windows PowerShell

# If not set, run setup:
python setup_gemini.py --api-key YOUR_KEY --save-env
```

### Issue: "google-generativeai not installed"

**Solution:**
```bash
pip install google-generativeai
# or
pip install -r requirements.txt
```

### Issue: "API key invalid"

**Solutions:**
1. Verify your API key at https://makersuite.google.com/app/apikey
2. Make sure there are no extra spaces or quotes
3. Try regenerating the API key
4. Check if the API key has proper permissions

### Issue: "quota exceeded"

**Solution:**
- Gemini has rate limits on the free tier
- Wait a few minutes and try again
- Consider upgrading to paid tier for higher limits
- Use `gemini-flash` for faster, lighter requests

### Issue: Gemini not being used even when available

**Solution:**
```python
# Make sure to set prefer_gemini=True
tester = PromptDiversityTester()
llm = tester.route_prompt_to_llm(prompt, prefer_gemini=True)
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or install individually:
pip install google-generativeai>=0.3.0
```

---

## Security Best Practices

1. **Never commit API keys to Git**
   ```bash
   # Make sure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables**
   - Don't hardcode API keys in code
   - Use .env files or system environment variables

3. **Rotate keys regularly**
   - Generate new keys periodically
   - Revoke old keys at Google AI Studio

4. **Set up usage alerts**
   - Monitor usage at Google Cloud Console
   - Set up billing alerts if on paid tier

---

## Performance Tips

1. **Choose the right model:**
   - Use `gemini-flash` for quick, simple tasks
   - Use `gemini-pro` for complex reasoning

2. **Optimize token usage:**
   - Use prompt optimization to reduce costs
   - Set appropriate `max_tokens` limits

3. **Batch requests when possible:**
   - Process multiple prompts together
   - Use async/parallel processing for multiple requests

4. **Cache responses:**
   - Store frequently used responses
   - Implement response caching for repeated prompts

---

## API Limits (Free Tier)

- **Rate limit:** 60 requests per minute
- **Daily limit:** Varies by region
- **Max tokens:** 30,720 per request (gemini-pro)

For higher limits, consider upgrading to a paid plan.

---

## Additional Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Python SDK Reference](https://ai.google.dev/api/python/google/generativeai)
- [Pricing Information](https://ai.google.dev/pricing)
- [API Key Management](https://makersuite.google.com/app/apikey)

---

## Getting Help

If you encounter issues:

1. Run the diagnostic test:
   ```bash
   python setup_gemini.py --test
   ```

2. Check the detailed test output:
   ```bash
   python test_gemini.py
   ```

3. Verify your setup:
   ```bash
   python -c "import os; print('API Key set:', bool(os.getenv('GEMINI_API_KEY')))"
   ```

4. Check installation:
   ```bash
   pip list | grep google-generativeai
   ```

---

## Next Steps

1. ✅ Set up your API key
2. ✅ Run tests to verify integration
3. ✅ Try the examples above
4. ✅ Integrate Gemini into your workflows
5. ✅ Compare performance with local models

Enjoy using Gemini in your text optimization framework! 🚀
