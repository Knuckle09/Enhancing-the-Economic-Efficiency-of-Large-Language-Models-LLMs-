"""
Test script for the RL Text Optimization Framework API
This script helps verify that all components are working correctly.
"""

import sys
import os
import time
import json
import requests
from datetime import datetime

def test_cli_mode():
    """Test the CLI mode of run.py"""
    print("🔍 Testing CLI Mode...")
    try:
        from run import process_prompt
        
        test_prompt = "Write a Python function to calculate factorial"
        print(f"📝 Test prompt: {test_prompt}")
        
        result = process_prompt(test_prompt)
        
        if result:
            print("✅ CLI mode test successful!")
            print(f"   Strategy: {result['strategy_used']}")
            print(f"   Reduction: {result['metrics']['reduction_percent']:.1f}%")
            print(f"   Similarity: {result['metrics']['similarity']:.3f}")
            print(f"   LLM: {result['selected_llm']}")
            return True
        else:
            print("❌ CLI mode test failed!")
            return False
            
    except Exception as e:
        print(f"❌ CLI mode test error: {e}")
        return False

def test_api_server(port=5000):
    """Test the API server endpoints"""
    print("🔍 Testing API Server...")
    base_url = f"http://localhost:{port}/api"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API server not accessible: {e}")
        print("   Make sure to start the server with: python api.py")
        return False
    
    try:
        test_data = {
            "prompt": "Calculate the derivative of x^2",
            "include_response": False
        }
        
        response = requests.post(
            f"{base_url}/process", 
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result['data']
                print("✅ Process endpoint working")
                print(f"   Strategy: {data['strategy_used']}")
                print(f"   Reduction: {data['token_reduction_percent']}%")
                print(f"   LLM: {data['selected_llm']}")
                print(f"   Processing time: {result['processing_time']}s")
                return True
            else:
                print(f"❌ Process failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Process endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Process endpoint error: {e}")
        return False

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("🔍 Checking Requirements...")
    
    required_files = [
        "run.py", "api.py", "rl_optimizer.py", "llm_efficiency_test.py",
        "prompt_diversity_test.py", "text_optimizer_ppo.zip",
        "requirements.txt", "frontend_demo.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
    
    required_modules = [
        "torch", "stable_baselines3", "gymnasium", "sentence_transformers",
        "transformers", "spacy", "pandas", "scikit_learn", "numpy"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module.replace('_', '-'))
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"⚠️  Missing Python modules: {missing_modules}")
        print("   Install with: pip install -r requirements.txt")
    else:
        print("✅ All required Python modules available")
    
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print("✅ Ollama is available")
            models = result.stdout
            required_models = ["codellama", "qwen2-math", "tinyllama"]
            for model in required_models:
                if model in models:
                    print(f"✅ Model {model} found")
                else:
                    print(f"⚠️  Model {model} not found - install with: ollama pull {model}")
        else:
            print("⚠️  Ollama not accessible")
    except Exception as e:
        print(f"⚠️  Could not check Ollama: {e}")
    
    return len(missing_files) == 0

def main():
    """Main test function"""
    print("="*60)
    print("🚀 RL Text Optimization Framework - Test Suite")
    print("="*60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests_passed = 0
    total_tests = 3
    
    if check_requirements():
        tests_passed += 1
    print()
    
    if test_cli_mode():
        tests_passed += 1
    print()
    
    print("🔍 API Server Test (optional - requires server to be running)")
    if test_api_server():
        tests_passed += 1
    else:
        print("   To test API: Start server with 'python api.py' then run this test again")
    print()
    
    print("="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 2:
        print("✅ Framework is ready for deployment!")
        if tests_passed == 3:
            print("🌐 API server is also working correctly")
        else:
            print("💡 To enable web interface, start the API server with: python api.py")
    else:
        print("❌ Framework needs attention before deployment")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())