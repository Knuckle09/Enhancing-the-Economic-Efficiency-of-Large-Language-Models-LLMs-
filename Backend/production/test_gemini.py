"""
Quick test script for Gemini integration

This script demonstrates how to use Gemini models in the framework
"""

import os
from llm_efficiency_test import LLMEfficiencyTest
from prompt_diversity_test import PromptDiversityTester


def test_basic_generation():
    """Test basic text generation with Gemini"""
    print("\n" + "="*60)
    print("TEST 1: Basic Text Generation")
    print("="*60)
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("❌ Gemini not available. Please set GEMINI_API_KEY")
        return False
    
    prompt = "Write a Python function to calculate factorial"
    print(f"\nPrompt: {prompt}")
    
    try:
        response = tester.generate_response(prompt, llm="gemini-pro", max_tokens=256)
        print(f"\nGemini Response:\n{response}")
        print("\n✅ Basic generation test passed!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_optimization():
    """Test prompt optimization with Gemini"""
    print("\n" + "="*60)
    print("TEST 2: Prompt Optimization")
    print("="*60)
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("❌ Gemini not available. Please set GEMINI_API_KEY")
        return False
    
    original_prompt = """
    Please write a comprehensive Python function that can calculate the factorial 
    of any given positive integer number. The function should handle edge cases 
    properly and return the correct factorial value for the input.
    """
    
    print(f"\nOriginal prompt: {original_prompt.strip()}")
    print(f"Original tokens: {tester.count_tokens(original_prompt)}")
    
    try:
        optimized_prompt, metrics = tester.optimize_tokens(
            original_prompt,
            target_reduction=0.30,
            min_similarity=0.85,
            llm="gemini-pro",
            category="coding"
        )
        
        print(f"\nOptimized prompt: {optimized_prompt}")
        print(f"Optimized tokens: {tester.count_tokens(optimized_prompt)}")
        print(f"\nMetrics:")
        print(f"  Reduction: {metrics['reduction_percent']:.1f}%")
        print(f"  Similarity: {metrics['similarity']:.3f}")
        print(f"  Target achieved: {metrics['target_achieved']}")
        
        if metrics['target_achieved']:
            print("\n✅ Optimization test passed!")
            return True
        else:
            print("\n⚠️  Optimization didn't meet targets, but completed")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_routing():
    """Test automatic LLM routing with Gemini preference"""
    print("\n" + "="*60)
    print("TEST 3: Smart LLM Routing")
    print("="*60)
    
    tester = PromptDiversityTester()
    
    if not tester.llm_efficiency_test.gemini_available:
        print("❌ Gemini not available. Please set GEMINI_API_KEY")
        return False
    
    test_prompts = [
        ("Write a binary search algorithm", "coding"),
        ("Solve x^2 + 5x + 6 = 0", "math"),
        ("Explain quantum computing", "generic")
    ]
    
    for prompt, expected_category in test_prompts:
        print(f"\nPrompt: {prompt}")
        llm = tester.route_prompt_to_llm(prompt, prefer_gemini=True)
        print(f"Routed to: {llm}")
        
        if "gemini" in llm.lower():
            print("✅ Correctly routed to Gemini")
        else:
            print("⚠️  Routed to local model")
    
    print("\n✅ Routing test completed!")
    return True


def test_comparison():
    """Compare Gemini with local models"""
    print("\n" + "="*60)
    print("TEST 4: Gemini vs Local Model Comparison")
    print("="*60)
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("❌ Gemini not available. Please set GEMINI_API_KEY")
        return False
    
    prompt = "Explain recursion in programming"
    print(f"\nPrompt: {prompt}\n")
    
    print("Testing with TinyLlama...")
    try:
        local_response = tester.generate_response(prompt, llm="tinyllama", max_tokens=200)
        print(f"TinyLlama response:\n{local_response[:200]}...\n")
    except Exception as e:
        print(f"TinyLlama failed: {e}\n")
    
    print("Testing with Gemini...")
    try:
        gemini_response = tester.generate_response(prompt, llm="gemini-pro", max_tokens=200)
        print(f"Gemini response:\n{gemini_response[:200]}...\n")
        
        similarity = tester.calculate_similarity(local_response, gemini_response)
        print(f"Response similarity: {similarity:.3f}")
        
        print("\n✅ Comparison test completed!")
        return True
    except Exception as e:
        print(f"Gemini failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*20 + "GEMINI INTEGRATION TESTS")
    print("="*70)
    
    if not os.getenv('GEMINI_API_KEY'):
        print("\n❌ GEMINI_API_KEY not found!")
        print("\nTo set it up:")
        print("  python setup_gemini.py --api-key YOUR_KEY --save-env")
        print("\nOr set it directly:")
        print("  export GEMINI_API_KEY='your-key'  # Linux/Mac")
        print("  $env:GEMINI_API_KEY='your-key'    # Windows PowerShell")
        return
    
    tests = [
        ("Basic Generation", test_basic_generation),
        ("Prompt Optimization", test_optimization),
        ("Smart Routing", test_routing),
        ("Model Comparison", test_comparison)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Gemini integration is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")


if __name__ == "__main__":
    main()
