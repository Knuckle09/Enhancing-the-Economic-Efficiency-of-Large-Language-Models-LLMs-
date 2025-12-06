"""
Simple example demonstrating Gemini integration

This shows the most common use cases for Gemini in the framework.
"""

import os
from llm_efficiency_test import LLMEfficiencyTest


def example_1_basic_usage():
    """Example 1: Basic text generation with Gemini"""
    print("\n" + "="*60)
    print("Example 1: Basic Gemini Usage")
    print("="*60 + "\n")
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("⚠️  Gemini not available. Set GEMINI_API_KEY first:")
        print("   python setup_gemini.py --api-key YOUR_KEY --save-env")
        return
    
    prompt = "Write a Python function to reverse a string"
    print(f"Prompt: {prompt}\n")
    
    response = tester.generate_response(prompt, llm="gemini-pro", max_tokens=200)
    print(f"Gemini Response:\n{response}\n")


def example_2_optimization():
    """Example 2: Optimize prompts before sending to Gemini"""
    print("\n" + "="*60)
    print("Example 2: Prompt Optimization + Gemini")
    print("="*60 + "\n")
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("⚠️  Gemini not available")
        return
    
    original = """
    Please write a comprehensive and detailed Python function that can
    efficiently reverse any given string. The function should handle all
    edge cases including empty strings, single character strings, and
    strings with special characters. Please include proper error handling.
    """
    
    print(f"Original prompt ({tester.count_tokens(original)} tokens):")
    print(f"{original.strip()}\n")
    
    optimized, metrics = tester.optimize_tokens(
        original,
        target_reduction=0.30,
        min_similarity=0.85,
        category="coding"
    )
    
    print(f"Optimized prompt ({tester.count_tokens(optimized)} tokens):")
    print(f"{optimized}\n")
    print(f"Token reduction: {metrics['reduction_percent']:.1f}%")
    print(f"Semantic similarity: {metrics['similarity']:.3f}\n")
    
    response = tester.generate_response(optimized, llm="gemini-pro", max_tokens=200)
    print(f"Gemini Response:\n{response}\n")


def example_3_smart_routing():
    """Example 3: Automatic LLM routing with Gemini preference"""
    print("\n" + "="*60)
    print("Example 3: Smart LLM Routing")
    print("="*60 + "\n")
    
    from prompt_diversity_test import PromptDiversityTester
    
    tester = PromptDiversityTester()
    
    if not tester.llm_efficiency_test.gemini_available:
        print("⚠️  Gemini not available")
        return
    
    prompts = [
        "Write a bubble sort algorithm in Python",
        "Solve the equation: 2x^2 + 5x - 3 = 0",
        "Explain the theory of relativity"
    ]
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        
        llm_local = tester.route_prompt_to_llm(prompt, prefer_gemini=False)
        print(f"  Without Gemini preference: {llm_local}")
        
        llm_gemini = tester.route_prompt_to_llm(prompt, prefer_gemini=True)
        print(f"  With Gemini preference: {llm_gemini}\n")


def example_4_comparison():
    """Example 4: Compare Gemini with local models"""
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60 + "\n")
    
    tester = LLMEfficiencyTest()
    
    if not tester.gemini_available:
        print("⚠️  Gemini not available")
        return
    
    prompt = "What is recursion in programming?"
    print(f"Prompt: {prompt}\n")
    
    models = ["tinyllama", "gemini-pro"]
    
    for model in models:
        print(f"--- {model.upper()} ---")
        try:
            response = tester.generate_response(prompt, llm=model, max_tokens=150)
            print(f"{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def example_5_error_handling():
    """Example 5: Proper error handling with fallback"""
    print("\n" + "="*60)
    print("Example 5: Error Handling & Fallback")
    print("="*60 + "\n")
    
    tester = LLMEfficiencyTest()
    
    prompt = "Explain machine learning"
    
    try:
        if tester.gemini_available:
            print("Trying Gemini Pro...")
            response = tester.generate_response(prompt, llm="gemini-pro", max_tokens=100)
            print(f"✅ Gemini Response:\n{response}\n")
        else:
            raise Exception("Gemini not available")
    except Exception as e:
        print(f"⚠️  Gemini failed: {e}")
        print("Falling back to TinyLlama...")
        response = tester.generate_response(prompt, llm="tinyllama", max_tokens=100)
        print(f"✅ TinyLlama Response:\n{response}\n")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*20 + "GEMINI USAGE EXAMPLES")
    print("="*70)
    
    if not os.getenv('GEMINI_API_KEY'):
        print("\n❌ GEMINI_API_KEY not set!")
        print("\nTo set it up:")
        print("  python setup_gemini.py --api-key YOUR_KEY --save-env")
        print("\nOr set manually:")
        print("  export GEMINI_API_KEY='your-key'  # Linux/Mac")
        print("  $env:GEMINI_API_KEY='your-key'    # Windows PowerShell")
        print("\nRunning examples with local models only...\n")
    
    examples = [
        example_1_basic_usage,
        example_2_optimization,
        example_3_smart_routing,
        example_4_comparison,
        example_5_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Example failed: {e}\n")
            continue
    
    print("\n" + "="*70)
    print(" "*25 + "EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Set your Gemini API key if not already set")
    print("  2. Try modifying these examples for your use case")
    print("  3. Check GEMINI_SETUP.md for detailed documentation")
    print("  4. Run test_gemini.py for comprehensive tests\n")


if __name__ == "__main__":
    main()
