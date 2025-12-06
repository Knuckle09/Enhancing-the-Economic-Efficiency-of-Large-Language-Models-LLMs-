"""
Complete API Test Suite for RL-Based Text Optimization Framework
Tests all endpoints and functionality
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing RL-Based Text Optimization API")
    print("=" * 60)
    
    print("1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health: {data['status']}")
            print(f"   ✅ Framework Initialized: {data['framework_initialized']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    print("\n2️⃣ Testing Status...")
    try:
        response = requests.get(f"{base_url}/api/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Initialized: {data['initialized']}")
            if data.get('model_path'):
                print(f"   ✅ Model: {data['model_path']}")
        else:
            print(f"   ❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Status error: {e}")
    
    print("\n3️⃣ Testing Strategies...")
    try:
        response = requests.get(f"{base_url}/api/strategies", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Strategies: {list(data['strategies'].keys())}")
            print(f"   ✅ Categories: {data['categories']}")
            print(f"   ✅ LLMs: {list(data['llms'].keys())}")
        else:
            print(f"   ❌ Strategies failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Strategies error: {e}")
    
    print("\n4️⃣ Testing Quick Processing...")
    test_prompts = [
        {"prompt": "Write a Python function to sort a list", "category": "coding"},
        {"prompt": "Calculate the derivative of x^2", "category": "math"}, 
        {"prompt": "Explain the benefits of renewable energy", "category": "generic"}
    ]
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n   Test 4.{i} - {test['category'].title()} Prompt:")
        try:
            payload = {
                "prompt": test["prompt"],
                "include_response": False
            }
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/process",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                result = data['data']
                print(f"     ✅ Category: {result['category']}")
                print(f"     ✅ Strategy: {result['strategy_used']}")
                print(f"     ✅ Reduction: {result['token_reduction_percent']}%")
                print(f"     ✅ Similarity: {result['similarity']}")
                print(f"     ✅ LLM: {result['selected_llm']}")
                print(f"     ✅ Time: {processing_time:.2f}s")
                print(f"     📝 Optimized: '{result['optimized_prompt']}'")
            else:
                print(f"     ❌ Failed: {response.status_code}")
                print(f"     ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    print("\n5️⃣ Testing Full Processing with LLM Response...")
    try:
        payload = {
            "prompt": "Calculate 3 + 4",
            "include_response": True
        }
        
        print("   🤖 Generating LLM response (this may take 10-30 seconds)...")
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/process",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            result = data['data']
            print(f"   ✅ Full processing completed in {processing_time:.2f}s")
            print(f"   ✅ Category: {result['category']}")
            print(f"   ✅ LLM: {result['selected_llm']}")
            print(f"   ✅ Reduction: {result['token_reduction_percent']}%")
            print(f"   📝 Response: {result['response'][:200]}...")
        else:
            print(f"   ❌ Full processing failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Full processing error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("\n💡 Frontend Usage Example:")
    print("   Open: http://localhost:8080/index.html")
    print("   API Base: http://localhost:5000/api/")
    print("\n🔗 Available Endpoints:")
    print("   GET  /api/health      - Health check")
    print("   GET  /api/status      - Framework status") 
    print("   GET  /api/strategies  - Available strategies")
    print("   POST /api/process     - Process prompts")
    return True

if __name__ == "__main__":
    test_api()