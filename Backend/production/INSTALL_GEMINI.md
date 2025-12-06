# Complete Installation Guide - Gemini Integration

This guide covers everything needed to install and set up Gemini in your framework.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection
- Google account (for API key)

---

## Step-by-Step Installation

### 1. Install Dependencies

#### Option A: Install All Requirements
```bash
pip install -r requirements.txt
```

#### Option B: Install Gemini Package Only
```bash
pip install google-generativeai
```

#### Verify Installation
```bash
python -c "import google.generativeai; print('✅ Installed')"
```

---

### 2. Get Gemini API Key

#### a) Visit Google AI Studio
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the API key (looks like: `AIzaSy...`)

#### b) Important Notes
- ⚠️ Keep your API key secure - treat it like a password
- ⚠️ Don't share it publicly or commit it to Git
- ⚠️ You can revoke and regenerate keys anytime

---

### 3. Configure API Key

Choose one of these methods:

#### Method 1: Using Setup Script (Recommended)
```bash
python setup_gemini.py --api-key YOUR_API_KEY_HERE --save-env
```

This will:
- Save key to `.env` file
- Add `.env` to `.gitignore`
- Set key for current session

#### Method 2: Manual Environment Variable

**Linux/Mac:**
```bash
# Temporary (current session)
export GEMINI_API_KEY='YOUR_API_KEY_HERE'

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo "export GEMINI_API_KEY='YOUR_API_KEY_HERE'" >> ~/.bashrc
source ~/.bashrc
```

**Windows PowerShell:**
```powershell
# Temporary
$env:GEMINI_API_KEY='YOUR_API_KEY_HERE'

# Permanent
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "YOUR_API_KEY_HERE", "User")
```

**Windows Command Prompt:**
```cmd
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
```

#### Method 3: .env File (Manual)
Create a file named `.env` in the project root:
```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Then install python-dotenv if needed:
```bash
pip install python-dotenv
```

---

### 4. Test Installation

#### Quick Test
```bash
python setup_gemini.py --test
```

Expected output:
```
🔍 Testing Gemini API connection...
✅ google-generativeai package is installed
✅ API key configured successfully

🧪 Testing with a simple prompt...
✅ Gemini API is working!

Test response: Hello! How can I help you today?...
```

#### Comprehensive Test
```bash
python test_gemini.py
```

This runs:
1. Basic generation test
2. Prompt optimization test
3. Smart routing test
4. Model comparison test

---

### 5. Verify Integration

#### Check in Python
```python
from llm_efficiency_test import LLMEfficiencyTest

tester = LLMEfficiencyTest()
print(f"Gemini available: {tester.gemini_available}")
```

Expected output:
```
✅ Gemini API initialized successfully
Gemini available: True
```

#### Check via API
```bash
# Start server
python api.py

# In another terminal
curl http://localhost:5000/api/strategies | python -m json.tool
```

Look for `"gemini_available": true` in the response.

---

## Troubleshooting Installation

### Issue 1: Package Installation Failed

**Problem:**
```
ERROR: Could not find a version that satisfies the requirement google-generativeai
```

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing again
pip install google-generativeai

# Or specify version
pip install google-generativeai>=0.3.0
```

### Issue 2: API Key Not Found

**Problem:**
```
⚠️ GEMINI_API_KEY not found in environment variables
```

**Solution:**
```bash
# Verify key is set
echo $GEMINI_API_KEY  # Linux/Mac
echo %GEMINI_API_KEY%  # Windows CMD
echo $env:GEMINI_API_KEY  # Windows PowerShell

# If empty, set it again
python setup_gemini.py --api-key YOUR_KEY --save-env
```

### Issue 3: Import Error

**Problem:**
```python
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution:**
```bash
# Check if installed
pip list | grep google-generativeai

# If not found, install
pip install google-generativeai

# Check Python environment
which python  # Linux/Mac
where python  # Windows

# Make sure you're using the right Python
python --version
```

### Issue 4: API Test Failed

**Problem:**
```
❌ Gemini API test failed: 400 API key not valid
```

**Solutions:**
1. **Check API key is correct:**
   - Visit https://makersuite.google.com/app/apikey
   - Copy the key carefully (no spaces or quotes)

2. **Regenerate API key:**
   - Create a new key at Google AI Studio
   - Update your configuration

3. **Check for extra characters:**
   ```bash
   # Print key to check for issues
   python -c "import os; print(repr(os.getenv('GEMINI_API_KEY')))"
   ```

### Issue 5: Permission Denied

**Problem:**
```
PermissionError: [Errno 13] Permission denied: '.env'
```

**Solution:**
```bash
# Check file permissions
ls -la .env  # Linux/Mac
icacls .env  # Windows

# Fix permissions
chmod 644 .env  # Linux/Mac

# Or delete and recreate
rm .env
python setup_gemini.py --api-key YOUR_KEY --save-env
```

### Issue 6: Network/Connection Issues

**Problem:**
```
ConnectionError: Failed to establish connection
```

**Solutions:**
1. **Check internet connection:**
   ```bash
   ping google.com
   ```

2. **Check firewall:**
   - Ensure Python can access the internet
   - Check corporate firewall/proxy settings

3. **Try with proxy:**
   ```python
   import os
   os.environ['HTTP_PROXY'] = 'http://proxy:port'
   os.environ['HTTPS_PROXY'] = 'http://proxy:port'
   ```

### Issue 7: Rate Limit Exceeded

**Problem:**
```
429 Resource exhausted: Quota exceeded
```

**Solutions:**
1. **Wait a few minutes** (free tier: 60 req/min)
2. **Use Gemini Flash** (faster, uses less quota)
3. **Upgrade to paid tier** for higher limits

---

## Verification Checklist

After installation, verify:

- [ ] Package installed: `pip list | grep google-generativeai`
- [ ] API key set: `python -c "import os; print(bool(os.getenv('GEMINI_API_KEY')))"`
- [ ] Connection works: `python setup_gemini.py --test`
- [ ] Tests pass: `python test_gemini.py`
- [ ] Examples run: `python examples_gemini.py`
- [ ] API detects it: `curl http://localhost:5000/api/strategies`

---

## Platform-Specific Notes

### Linux/Ubuntu
```bash
# May need these system packages
sudo apt-get install python3-dev build-essential

# Virtual environment recommended
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### macOS
```bash
# Install via Homebrew Python
brew install python3

# Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```powershell
# Use PowerShell (not CMD)
# Virtual environment
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Docker
```dockerfile
# Add to your Dockerfile
RUN pip install google-generativeai

# Set environment variable
ENV GEMINI_API_KEY=your_key_here
```

---

## Post-Installation

### Run Examples
```bash
python examples_gemini.py
```

### Try CLI
```bash
python run.py "Your prompt here"
```

### Start API Server
```bash
python api.py
```

### Check Documentation
- Quick Reference: `GEMINI_QUICKSTART.md`
- Full Guide: `GEMINI_SETUP.md`
- Summary: `GEMINI_SUMMARY.md`

---

## Updating

### Update Package
```bash
pip install --upgrade google-generativeai
```

### Update Framework
```bash
git pull  # If using Git
# Or download latest files
```

### Regenerate API Key
If you need a new key:
1. Visit https://makersuite.google.com/app/apikey
2. Revoke old key
3. Create new key
4. Update configuration:
   ```bash
   python setup_gemini.py --api-key NEW_KEY --save-env
   ```

---

## Uninstalling

### Remove Package
```bash
pip uninstall google-generativeai
```

### Remove API Key
```bash
# Delete .env file
rm .env

# Unset environment variable
unset GEMINI_API_KEY  # Linux/Mac

# Windows PowerShell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $null, "User")
```

### Revert Code Changes
The framework will automatically fall back to local models if Gemini is not available. No code changes needed!

---

## Getting Help

### Run Diagnostics
```bash
python setup_gemini.py --test
python test_gemini.py
```

### Check Logs
```bash
# API server logs
python api.py  # Look for "Gemini API initialized"

# Python test
python -c "from llm_efficiency_test import LLMEfficiencyTest; LLMEfficiencyTest()"
```

### Common Commands
```bash
# Check installation
pip show google-generativeai

# Check version
python -c "import google.generativeai as genai; print(genai.__version__)"

# Check API key
python -c "import os; print('Set' if os.getenv('GEMINI_API_KEY') else 'Not set')"

# Test import
python -c "import google.generativeai; print('OK')"
```

---

## Resources

- **Setup Script:** `python setup_gemini.py --help`
- **Quick Guide:** [GEMINI_QUICKSTART.md](GEMINI_QUICKSTART.md)
- **Full Guide:** [GEMINI_SETUP.md](GEMINI_SETUP.md)
- **API Key:** https://makersuite.google.com/app/apikey
- **Gemini Docs:** https://ai.google.dev/docs
- **Python SDK:** https://ai.google.dev/api/python

---

## Next Steps

1. ✅ Complete installation
2. ✅ Test connection
3. ✅ Run examples
4. ✅ Read documentation
5. ✅ Integrate into your code

Enjoy using Gemini! 🚀
