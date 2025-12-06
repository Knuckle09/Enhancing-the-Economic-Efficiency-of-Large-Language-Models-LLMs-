"""
Complete End-to-End Text Optimization Framework

Framework Goal:
1. Take a prompt as input
2. Auto-classify prompt category using NLP patterns
3. Apply RL-based token reduction (30%+ with 80%+ semantic meaning)
4. Select suitable LLM based on auto-detected category
5. Generate response from selected LLM

Usage: python run.py [prompt]
"""

import os
import sys
import subprocess
from stable_baselines3 import PPO
from rl_optimizer import RLOptimizer, get_latest_training_data_file, DEFAULT_TRAINING_TIMESTEPS
from prompt_diversity_test import PromptDiversityTester

def process_prompt(prompt):
    """
    Complete framework pipeline: Input → RL Token Reduction → LLM Selection → Response Generation
    
    Args:
        prompt (str): Input prompt to process
        
    Returns:
        dict: Complete results including optimized prompt, selected LLM, and response
    """
    print("=" * 80)
    print("🚀 FRAMEWORK: TOKEN REDUCTION & LLM RESPONSE GENERATION")
    print("=" * 80)
    print(f"📝 Input Prompt: {prompt}")
    print("-" * 80)

    try:
        training_data_file = get_latest_training_data_file("./results")
        rl_optimizer = RLOptimizer(training_data_file)
        
        prompt_tester = PromptDiversityTester()
        
        category = prompt_tester.classify_prompt(prompt)
        print(f"📂 Auto-detected Category: {category}")
        
        model_path = "./models/text_optimizer_ppo.zip"
        if os.path.exists(model_path):
            print(f"📦 Loading trained RL model: {model_path}")
            rl_optimizer.model = PPO.load(model_path)
        else:
            print("⚠️ No trained model found. Training new model...")
            rl_optimizer.train(total_timesteps=DEFAULT_TRAINING_TIMESTEPS, save_path=model_path)
        
        print(f"🎯 Processing prompt: {prompt}...")
        action, strategy = rl_optimizer.predict_optimal_strategy(prompt, category)
        optimized_prompt, metrics = rl_optimizer.env.apply_optimization_strategy(prompt, action)
        
        print(f"📊 Token Reduction: {metrics['reduction_percent']:.1f}%")
        print(f"📊 Similarity: {metrics['similarity']:.3f}")
        print(f"📊 Strategy Used: {strategy}")
        
        if metrics['reduction_percent'] < 25:
            print("⚠️  Warning: Token reduction below 25% target")
        if metrics['similarity'] < 0.8:
            print("⚠️  Warning: Similarity below 80% target")
            
        selected_llm = prompt_tester.route_prompt_to_llm(optimized_prompt)
        print(f"🤖 Selected LLM: {selected_llm}")
        
        try:
            response = subprocess.check_output(
                ["ollama", "run", selected_llm, optimized_prompt], 
                text=True, timeout=120
            ).strip()
            
            result = {
                'original_prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'selected_llm': selected_llm,
                'response': response,
                'metrics': metrics,
                'strategy_used': strategy
            }
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            result = {
                'original_prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'selected_llm': selected_llm,
                'response': f"Error generating response: {e}",
                'metrics': metrics,
                'strategy_used': strategy
            }
        print("\n" + "=" * 80)
        print("✅ FRAMEWORK RESULTS:")
        print("=" * 80)
        print(f"🎯 Optimized Prompt: {result['optimized_prompt']}")
        print(f"🤖 Selected LLM: {result['selected_llm']}")
        print(f"📊 Strategy: {result['strategy_used']}")
        print(f"📈 Token Reduction: {result['metrics']['reduction_percent']:.1f}%")
        print(f"🎭 Similarity: {result['metrics']['similarity']:.3f}")
        print(f"✅ Target Met: {'Yes' if result['metrics']['target_achieved'] else 'No'}")
        print("-" * 80)
        print("🤖 LLM RESPONSE:")
        print(result['response'])
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"❌ Framework failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the framework."""
    if len(sys.argv) >= 2:
        prompt = sys.argv[1]
        process_prompt(prompt)
    else:
        print("🎯 INTERACTIVE MODE - Testing Framework with Sample Prompts")
        print("=" * 80)
        
        sample_prompts = [
            "Calculate the area under the curve y = x^2 from x = 0 to x = 4 using integration",
            "Write a Python function to implement quicksort algorithm with error handling",
            "Design a marketing strategy for launching a new smartphone targeting millennials"
        ]
        
        for i, prompt in enumerate(sample_prompts, 1):
            print(f"\n🔄 TESTING PROMPT {i}/3:")
            result = process_prompt(prompt)
            
            if result:
                print(f"✅ Test {i} completed successfully!")
            else:
                print(f"❌ Test {i} failed!")
            
            print("\n" + "="*80)

if __name__ == "__main__":
    main()
