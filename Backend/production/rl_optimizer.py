"""
Reinforcement Learning Environment for Text Preprocessing Optimization
Uses PPO (Proximal Policy Optimization) to learn optimal preprocessing strategies

Author: Samarth & team
Requirements: pip install stable-baselines3 gym
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import json
import os
import subprocess
from llm_efficiency_test import LLMEfficiencyTest
from prompt_diversity_test import PromptDiversityTester
import psutil
import time
import glob

import sys
import io
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    else:
        import os
        new_out = io.TextIOWrapper(os.fdopen(sys.stdout.fileno(), 'wb', 0), encoding=sys.stdout.encoding, line_buffering=True)
        new_err = io.TextIOWrapper(os.fdopen(sys.stderr.fileno(), 'wb', 0), encoding=sys.stderr.encoding, line_buffering=True)
        sys.stdout = new_out
        sys.stderr = new_err
except Exception:
    pass

TRAINING_TIMESTEPS = 0

REQUIRED_MODELS = [
    "llava:latest",
    "moondream:latest",
    "qwen2-math:latest",
    "qwen2.5-coder:3b",
    "phi3:latest",
    "t1c/deepseek-math-7b-rl:latest",
    "codellama:latest",
    "codellama:7b",
    "tinyllama:latest"
]

DEFAULT_TRAINING_TIMESTEPS = 1000

def check_ollama_running():
    try:
        subprocess.check_output(["ollama", "list"])
        return True
    except Exception:
        return False

def get_installed_models():
    try:
        output = subprocess.check_output(["ollama", "list"], text=True)
        return [line.split()[0] for line in output.strip().splitlines() if line]
    except Exception:
        return []

def ensure_models_available():
    """Ensure all required Ollama models are available."""
    if not check_ollama_running():
        print("❌ Ollama is not running. Please start Ollama server first (run `ollama serve` in a terminal).")
        exit(1)
    installed = get_installed_models()
    for model in REQUIRED_MODELS:
        base_model = model.split(":")[0]
        if not any(base_model in m for m in installed):
            print(f"🔄 Pulling missing model: {model}")
            subprocess.run(["ollama", "pull", model])
        else:
            print(f"✅ Model available: {model}")


class TextOptimizationEnv(gym.Env):
    """
    RL Environment for learning text preprocessing optimization
    
    State: Text features (length, domain, complexity)
    Action: Optimization strategy (conservative, balanced, aggressive) + LLM choice
    Reward: Composite score (token_reduction * 0.4 + similarity * 0.6)
    """
    
    def __init__(self, training_data_file=None):
        super(TextOptimizationEnv, self).__init__()
        raw_data = self.load_training_data_full(training_data_file)

        normalized = []
        for item in (raw_data or []):
            if isinstance(item, dict):
                if 'state' in item and isinstance(item['state'], dict):
                    sample = item['state']
                else:
                    sample = item

                prompt = sample.get('original_prompt', sample.get('optimized_prompt', sample.get('prompt', '')))
                category = sample.get('category', sample.get('state', {}).get('category', sample.get('category', 'generic')))

                if isinstance(prompt, str) and prompt.strip():
                    normalized.append({'original_prompt': prompt, 'category': category, **sample})
            else:
                continue

        if len(normalized) != len(raw_data or []):
            removed = (len(raw_data or []) - len(normalized))
            if removed > 0:
                print(f"[ENV] Filtered out {removed} empty or malformed training samples")

        self.training_data = normalized
        self.current_idx = 0
        self.current_state = None
        
        self.llm_tester = LLMEfficiencyTest()

        self.data_stats = self._analyze_training_data()

        self.categories = ['coding', 'generic', 'math']

        self.category2idx = {cat: i for i, cat in enumerate(self.categories)}

        self.strategies = ["conservative", "balanced", "aggressive"]

        self.action_space = spaces.Discrete(len(self.strategies))
        state_size = 2 + len(self.categories)
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_size,), dtype=float)
        self.episode_count = 0
        self.max_episodes = len(self.training_data) if self.training_data else 100

    def _analyze_training_data(self):
        """Analyze training data distribution for better RL training"""
        if not self.training_data:
            return {'total': 0, 'categories': {}, 'llms': {}, 'success_rate': 0}
        
        stats = {
            'total': len(self.training_data),
            'categories': {},
            'llms': {},
            'strategies': {},
            'success_rate': 0,
            'avg_reduction': 0,
            'avg_similarity': 0
        }
        
        successful_count = 0
        total_reduction = 0
        total_similarity = 0
        
        for sample in self.training_data:
            state = sample.get('state', sample)
            
            category = state.get('category', 'generic')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            strategy = state.get('strategy', sample.get('strategy_used', 'unknown'))
            stats['strategies'][strategy] = stats['strategies'].get(strategy, 0) + 1
            
            explicit_success = None
            if 'rl_ready' in sample:
                explicit_success = bool(sample.get('rl_ready'))
            elif 'target_achieved' in sample:
                explicit_success = bool(sample.get('target_achieved'))

            reduction_val = state.get('reduction_percent', state.get('token_reduction', None))
            similarity_val = state.get('similarity', state.get('semantic_similarity', None))

            if explicit_success is True:
                successful_count += 1
            else:
                try:
                    if reduction_val is not None and similarity_val is not None:
                        if float(reduction_val) >= 30.0 and float(similarity_val) >= 0.80:
                            successful_count += 1
                except Exception:
                    pass

            if reduction_val is not None:
                try:
                    total_reduction += float(reduction_val)
                except Exception:
                    total_reduction += 0
            else:
                total_reduction += 0

            if similarity_val is not None:
                try:
                    total_similarity += float(similarity_val)
                except Exception:
                    total_similarity += 0
            else:
                total_similarity += 0
        
        stats['success_rate'] = successful_count / stats['total'] * 100
        stats['avg_reduction'] = total_reduction / stats['total']
        stats['avg_similarity'] = total_similarity / stats['total']
        
        print(f"📊 Training Data Analysis:")
        print(f"   Total samples: {stats['total']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Categories: {dict(stats['categories'])}")
        print(f"   LLMs: {dict(stats['llms'])}")
        print(f"   Avg reduction: {stats['avg_reduction']:.1f}%")
        print(f"   Avg similarity: {stats['avg_similarity']:.3f}")
        
        return stats

    def load_training_data_full(self, data_file):
        if data_file and os.path.exists(data_file):
            with open(data_file, 'r') as f:
                return json.load(f)
        return []

    def extract_features(self, state):
        """
        Extract enhanced features for better RL training
        State format: {'original_prompt': ..., 'category': ..., 'original_tokens': ...}
        Returns: [normalized_length, complexity, category_one_hot...]
        """
        prompt = state['original_prompt']
        category = state.get('category', 'generic')

        words = prompt.split()
        normalized_length = min(len(words) / 100, 1.0)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        complexity = min(avg_word_length / 10, 1.0)

        category_vec = [1 if category == cat else 0 for cat in self.categories]

        features = [normalized_length, complexity] + category_vec

        return np.array(features, dtype=np.float32)

    def apply_optimization_strategy(self, prompt, action):
        """Map RL actions to strategies and delegate optimization to LLMEfficiencyTest."""
        try:
            try:
                idx = int(action)
            except Exception:
                idx = 1

            if idx < 0 or idx >= len(self.strategies):
                print(f"⚠️ Invalid action {action}; defaulting to 'balanced'")
                idx = 1

            strategy_name = self.strategies[idx]

            strategy_map = {
                'conservative': (0.15, 0.90),
                'balanced': (0.30, 0.85),
                'aggressive': (0.35, 0.85)
            }

            target_reduction, min_similarity = strategy_map.get(strategy_name, (0.30, 0.85))

            category = self.current_state.get('category') if self.current_state else None

            optimized_prompt, metrics = self.llm_tester.optimize_tokens(
                prompt,
                target_reduction=target_reduction,
                min_similarity=min_similarity,
                category=category
            )

            reduction = metrics.get('reduction_percent', metrics.get('token_reduction', 0))
            similarity = metrics.get('similarity', metrics.get('semantic_similarity', metrics.get('similarity', 0)))
            metrics['reduction_percent'] = reduction
            metrics['similarity'] = similarity

            metrics['target_achieved'] = bool(reduction >= 30 and similarity >= 0.8)

            return optimized_prompt, metrics
        except Exception as e:
            print(f"Optimization failed: {e}")
            fallback_similarity = 0.8

            return prompt, {
                'reduction_percent': 0,
                'similarity': float(fallback_similarity),
                'target_achieved': False
            }
    
    def calculate_reward(self, metrics, target_similarity=0.85):
        """Calculate composite reward for RL training"""
        reduction_score = metrics['reduction_percent'] / 100
        similarity_score = metrics['similarity']
        
        similarity_penalty = max(0, target_similarity - similarity_score) * 2
        
        reward = (reduction_score * 0.4 + similarity_score * 0.6) - similarity_penalty
        
        if metrics.get('target_achieved', False):
            reward += 0.2
        
        return np.clip(reward, -1.0, 1.0)
    
    def reset(self, seed=None, options=None):
        """Reset environment to start new episode with optional seed."""
        if seed is not None:
            np.random.seed(seed)
        
        if self.training_data:
            sample = None
            for _ in range(10):
                candidate = self.training_data[np.random.randint(0, len(self.training_data))]
                prompt_text = (candidate.get('original_prompt', candidate.get('prompt', '')) if isinstance(candidate, dict) else '')
                if isinstance(prompt_text, str) and prompt_text.strip():
                    sample = candidate
                    break

            if sample is None:
                for candidate in self.training_data:
                    prompt_text = (candidate.get('original_prompt', candidate.get('prompt', '')) if isinstance(candidate, dict) else '')
                    if isinstance(prompt_text, str) and prompt_text.strip():
                        sample = candidate
                        break

            if sample is not None:
                self.current_state = {
                    'original_prompt': sample.get('original_prompt', sample.get('prompt', '')),
                    'category': sample.get('category', sample.get('state', {}).get('category', 'generic')),
                    'original_tokens': sample.get('original_tokens', len(sample.get('original_prompt', '').split()))
                }
            else:
                self.current_state = {
                    'original_prompt': "Default test prompt for optimization",
                    'category': 'generic',
                    'original_tokens': 6
                }
        else:
            self.current_state = {
                'original_prompt': "Default test prompt for optimization",
                'category': 'generic',
                'original_tokens': 6
            }
        
        observation = self.extract_features(self.current_state)
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        prompt = self.current_state['original_prompt']
        category = self.current_state.get('category', 'generic')

        optimized_prompt, metrics = self.apply_optimization_strategy(prompt, action)

        reward = self.calculate_reward(metrics)

        terminated = True
        truncated = False

        next_state = self.extract_features({
            'original_prompt': optimized_prompt,
            'category': category
        })

        info = {
            'original_prompt': prompt,
            'optimized_prompt': optimized_prompt,
            'metrics': metrics,
            'action_taken': action,
            'category': category
        }
        try:
            global TRAINING_TIMESTEPS
            ts_info = f"Timesteps: {TRAINING_TIMESTEPS}"
        except Exception:
            ts_info = "Timesteps: ?"

        print(f"[ENV] Step: Action {action}, Reward {reward:.3f} | {ts_info}")
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        print(f"Current prompt: {self.current_prompt[:100]}...")

    def set_max_episodes(self, max_episodes):
        """Set the maximum number of episodes for the environment."""
        self.max_episodes = max_episodes

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_calls = 0

    def _on_rollout_end(self) -> None:
        self.n_calls += 1
        print(f"[RL] Rollout {self.n_calls} finished. Timesteps so far: {self.num_timesteps}")
        try:
            global TRAINING_TIMESTEPS
            TRAINING_TIMESTEPS = int(self.num_timesteps)
        except Exception:
            pass

    def _on_step(self) -> bool:
        try:
            global TRAINING_TIMESTEPS
            TRAINING_TIMESTEPS = int(self.num_timesteps)
        except Exception:
            pass

        return True


class TimestepsLimitCallback(BaseCallback):
    """Stop training when a maximum number of timesteps is reached."""
    def __init__(self, max_timesteps, verbose=0):
        super().__init__(verbose)
        self.max_timesteps = int(max_timesteps)

    def _on_step(self) -> bool:
        try:
            if int(self.num_timesteps) >= self.max_timesteps:
                print(f"[RL] Reached requested max_timesteps={self.max_timesteps}. Stopping training.")
                return False
        except Exception:
            pass
        return True

class RLOptimizer:
    """Main RL trainer for text optimization"""
    
    def __init__(self, training_data_file=None):
        self.env = TextOptimizationEnv(training_data_file)
        self.model = None
        
        self.action_to_strategy = {0: "conservative", 1: "balanced", 2: "aggressive"}
        
        if not torch.cuda.is_available():
            print("⚠️ GPU is not available. Falling back to CPU. Training on CPU may be slow.")
            self.device = "cpu"
        else:
            self.device = "cuda"

        print(f"🎯 RL Training on: {self.device}")
        
    def train(self, total_timesteps=1000, save_path="./models/text_optimizer_ppo"):
        print(f"🚀 Starting RL Training on {self.device}...")

        vec_env = DummyVecEnv([lambda: self.env])

        policy_kwargs = dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU
        )

        self.model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=2,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            gamma=0.99,
            clip_range=0.2,
            tensorboard_log="./tensorboard_logs/",
            policy_kwargs=policy_kwargs,
            device=self.device
        )

        eval_env = DummyVecEnv([lambda: self.env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path="./eval_logs/",
            eval_freq=500,
            deterministic=True,
            render=False
        )

        progress_callback = ProgressCallback()
        timelimit_cb = TimestepsLimitCallback(total_timesteps)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, progress_callback, timelimit_cb],
            tb_log_name="text_optimization_ppo"
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)

        torch.cuda.empty_cache()

        print(f"🏆 Model saved to {save_path}")
        return self.model
    
    def extract_features(self, prompt, category="generic"):
        """Extract features from a prompt for RL model input."""
        state = {
            'original_prompt': prompt,
            'category': category,
            'original_tokens': len(prompt.split())
        }
        return self.env.extract_features(state)
    
    def predict_optimal_strategy(self, prompt, category="generic"):
        """
        Predict the optimal strategy for a given prompt using the RL model.
        This method only selects optimization strategies, not LLMs.
        """
        if self.model is None:
            print("❌ Model not trained. Using default balanced strategy.")
            return 1, "balanced"
            
        state = self.extract_features(prompt, category)

        action_raw, _ = self.model.predict(state, deterministic=True)

        try:
            action = int(np.asarray(action_raw).item())
        except Exception:
            action = int(action_raw)

        strategy = self.action_to_strategy.get(action, "balanced")

        return action, strategy

def monitor_gpu_usage():
    """Monitor GPU usage during processing"""
    if not torch.cuda.is_available():
        print("❌ No GPU available")
        return
    
    print("🖥️  GPU Monitoring Started")
    print("=" * 50)
    
    while True:
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_cached = torch.cuda.memory_reserved() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        print(f"GPU Memory: {gpu_memory:.2f}/{gpu_total:.2f} GB ({gpu_memory/gpu_total*100:.1f}%)")
        print(f"GPU Cached: {gpu_cached:.2f} GB")
        print(f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%")
        print("-" * 30)
        
        time.sleep(2)

def get_latest_training_data_file(results_dir="./results", prefer_multi_llm=True):
    """
    Return the path to the most recent training data file.
    Prioritizes multi-LLM files if prefer_multi_llm=True for better diversity.
    """
    if prefer_multi_llm:
        multi_llm_files = glob.glob(os.path.join(results_dir, "multi_llm_training_data_*.json"))
        if multi_llm_files:
            latest_multi_file = max(multi_llm_files, key=os.path.getmtime)
            print(f"🎯 Using multi-LLM training data: {latest_multi_file}")
            return latest_multi_file
    
    single_llm_files = glob.glob(os.path.join(results_dir, "rl_training_data_*.json"))
    if single_llm_files:
        latest_single_file = max(single_llm_files, key=os.path.getmtime)
        print(f"⚠️  Using single-LLM training data: {latest_single_file}")
        return latest_single_file
    
    raise FileNotFoundError(f"No training data files found in {results_dir}. Run data collection first.")

def main():
    """Main RL training pipeline - NOTE: For full framework use run.py"""
    print("="*80)
    print("RL MODEL TRAINING")
    print("="*80)
    print("📝 Note: This trains the RL model. For the complete framework, use: python run.py")
    print("="*80)

    ensure_models_available()

    print("\nTraining RL model...")
    trainer = RLOptimizer(get_latest_training_data_file("./results"))
    model = trainer.train(total_timesteps=DEFAULT_TRAINING_TIMESTEPS)
    
    print(f"\n✅ RL Model training completed!")
    print(f"🎯 Model saved to: ./models/text_optimizer_ppo.zip")
    print(f"📋 To use the complete framework: python run.py \"your prompt here\"")
    print("="*80)

if __name__ == "__main__":
    latest_training_file = get_latest_training_data_file("./results")
    print(f"Using training data: {latest_training_file}")
    trainer = RLOptimizer(latest_training_file)
    model = trainer.train(total_timesteps=DEFAULT_TRAINING_TIMESTEPS)