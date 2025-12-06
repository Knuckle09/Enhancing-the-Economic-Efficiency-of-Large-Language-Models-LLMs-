
import time
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from txtprprc1 import TextPreprocessor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import List, Dict, Any
import gc
from contextlib import contextmanager
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from sklearn.metrics.pairwise import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.cudnn.benchmark = True

class GPUManager:
    
    @staticmethod
    def get_device_info():
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return device, gpu_name, gpu_memory
        return "cpu", "CPU", 0
    
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return allocated, reserved
        return 0, 0

class NLPOptimizer:
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.available = False
        
        try:
            print(f"🔄 Loading NLP models on {self.device}...")
            
            self._load_models()
            self.available = True
            
            print(f"✅ NLP models loaded successfully on {self.device}")
            if self.device == "cuda":
                allocated, reserved = GPUManager.get_memory_usage()
                print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                
        except Exception as e:
            print(f"❌ NLP models failed to load: {e}")
            self._fallback_to_cpu()
    
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_models(self):
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1,
            model_kwargs=model_kwargs,
            framework="pt"
        )
        
        self.sentence_model = SentenceTransformer(
            'all-MiniLM-L6-v2', 
            device=self.device
        )
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("❌ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            raise
    
    def _fallback_to_cpu(self):
        print("🔄 Falling back to CPU...")
        self.device = "cpu"
        try:
            self._load_models()
            self.available = True
            print("✅ NLP models loaded on CPU")
        except Exception as e:
            print(f"❌ Complete failure to load models: {e}")
            self.available = False
    
    def smart_summarization(self, text, target_reduction=0.3, min_similarity=0.8):
        if not self.available or len(text.split()) < 20:
            return text
        
        original_length = len(text.split())
        target_length = max(int(original_length * (1 - target_reduction)), 10)
        
        try:
            with torch.amp.autocast('cuda') if self.device == "cuda" else torch.no_grad():
                attempts = [
                    (target_length, int(target_length * 0.8)),
                    (int(target_length * 1.2), int(target_length)),
                    (int(target_length * 1.4), int(target_length * 1.1))
                ]
                
                for max_len, min_len in attempts:
                    summary_params = {
                        "max_length": min(max_len, 1024),
                        "min_length": min(min_len, 5),
                        "do_sample": False,
                        "truncation": True,
                        "clean_up_tokenization_spaces": True
                    }
                    
                    if self.device == "cuda":
                        summary_params["batch_size"] = 8
                    
                    summary = self.summarizer(text, **summary_params)[0]['summary_text']
                    
                    similarity = self._calculate_similarity_gpu(text, summary)
                    if original_length > 0:
                        reduction = (original_length - len(summary.split())) / original_length
                    else:
                        reduction = 0.0
                    
                    if similarity >= min_similarity and reduction >= (target_reduction - 0.05):
                        return summary
                        
            return self.dependency_based_optimization(text, target_reduction, min_similarity)
            
        except Exception as e:
            print(f"GPU Summarization failed: {e}")
            GPUManager.clear_cache()
            return self.dependency_based_optimization(text, target_reduction, min_similarity)
    
    def dependency_based_optimization(self, text, target_reduction=0.3, min_similarity=0.8):
        if not self.available:
            return text
            
        try:
            doc = self.nlp(text)
            sentences = [sent for sent in doc.sents]
            
            if len(sentences) <= 1:
                return self._optimize_single_sentence(text, target_reduction, min_similarity)
            
            sentence_scores = []
            for sent in sentences:
                score = self._calculate_sentence_importance(sent, doc)
                sentence_scores.append((sent.text.strip(), score))
            
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            target_words = int(len(text.split()) * (1 - target_reduction))
            selected_sentences = []
            current_words = 0
            
            for sent_text, score in sentence_scores:
                sent_words = len(sent_text.split())
                if current_words + sent_words <= target_words:
                    selected_sentences.append(sent_text)
                    current_words += sent_words
                elif len(selected_sentences) == 0:
                    selected_sentences.append(sent_text)
                    break
            
            result = ' '.join(selected_sentences)
            
            similarity = self._calculate_similarity_gpu(text, result)
            if similarity < min_similarity and len(sentence_scores) > len(selected_sentences):
                next_sent = sentence_scores[len(selected_sentences)][0]
                result = ' '.join(selected_sentences + [next_sent])
            
            return result
            
        except Exception as e:
            print(f"Dependency optimization failed: {e}")
            return text
    
    def _optimize_single_sentence(self, text, target_reduction=0.3, min_similarity=0.8):
        if not self.available:
            return text
            
        try:
            doc = self.nlp(text)
            
            core_tokens = []
            important_deps = {'ROOT', 'nsubj', 'nsubjpass', 'dobj', 'pobj', 'compound', 'amod'}
            
            for token in doc:
                keep = False
                
                if token.dep_ in important_deps:
                    keep = True
                
                if token.ent_type_:
                    keep = True
                
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'NUM'] and not token.is_stop:
                    keep = True
                
                if token.like_num or not token.is_alpha:
                    keep = True
                
                if keep:
                    core_tokens.append(token.text)
            
            result = ' '.join(core_tokens)
            
            similarity = self._calculate_similarity_gpu(text, result)
            if similarity < min_similarity:
                for token in doc:
                    if (token.dep_ in ['det', 'prep', 'aux'] and 
                        token.text not in result.split()):
                        result = result.replace(token.head.text, 
                                              f"{token.text} {token.head.text}")
                        break
            
            return result if result.strip() else text
            
        except Exception as e:
            print(f"Single sentence optimization failed: {e}")
            return text
    
    def _calculate_sentence_importance(self, sentence, full_doc):
        try:
            score = 0
            
            entities = [ent for ent in sentence.ents]
            score += len(entities) * 0.3
            
            content_words = [token for token in sentence if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
            score += len(content_words) * 0.1
            
            sentences = list(full_doc.sents)
            sent_idx = sentences.index(sentence)
            if sent_idx == 0 or sent_idx == len(sentences) - 1:
                score += 0.2
            
            score = score / max(len(list(sentence)), 1)
            
            return score
            
        except Exception:
            return 0.1
    
    def _calculate_similarity_gpu(self, text1, text2):
        if not self.available:
            return 0.8
            
        try:
            with torch.no_grad():
                embeddings = self.sentence_model.encode(
                    [text1, text2],
                    convert_to_tensor=True,
                    device=self.device,
                    batch_size=32 if self.device == "cuda" else 8,
                    show_progress_bar=False
                )
                
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0),
                    embeddings[1].unsqueeze(0),
                    dim=1
                ).item()
                
                return float(similarity)
                
        except Exception as e:
            print(f"GPU similarity calculation failed: {e}")
            GPUManager.clear_cache()
            return 0.8

class LLMEfficiencyTest:

    def __init__(self, output_folder="./test_results", semantic_model_name="all-MiniLM-L6-v2", **kwargs):
        
        self.device, gpu_name, gpu_memory = GPUManager.get_device_info()
        print(f"🚀 Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {gpu_memory:.1f} GB")
        
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
        
        try:
            self.semantic_model = SentenceTransformer(semantic_model_name, device=self.device)
            print(f"✅ Semantic model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load semantic model on GPU, using CPU: {e}")
            self.semantic_model = SentenceTransformer(semantic_model_name, device="cpu")
        
        self.preprocessor = TextPreprocessor()
        
        self._preprocessed_cache = {}
        self._embedding_cache = {}
        
        self.nlp_optimizer = NLPOptimizer(device=self.device)
        
        self.llm_map = {
            "tinyllama": {"type": "ollama", "model": "tinyllama:latest"},
            "phi": {"type": "ollama", "model": "phi3:latest"},
            "codellama": {"type": "ollama", "model": "codellama:7b"},
            "qwen-coder": {"type": "ollama", "model": "qwen2.5-coder:3b"},
            "deepseek-math": {"type": "ollama", "model": "t1c/deepseek-math-7b-rl:latest"},
            "qwen-math": {"type": "ollama", "model": "qwen2-math:latest"},
            "moondream": {"type": "ollama", "model": "moondream:latest"},
            "llava": {"type": "ollama", "model": "llava:latest"},
            "gemini": {"type": "gemini", "model": "gemini-2.5-flash"},
            "gemini-pro": {"type": "gemini", "model": "gemini-2.5-pro"},
            "gemini-flash": {"type": "gemini", "model": "gemini-2.5-flash"}
        }

        self.config = {
            'max_retries': kwargs.get('max_retries', 3),
            'timeout': kwargs.get('timeout', 60),
            'batch_size': kwargs.get('batch_size', 8 if self.device == "cuda" else 2),
            'cache_size': kwargs.get('cache_size', 1000),
            'enable_fallbacks': kwargs.get('enable_fallbacks', True)
        }
        
        self.ollama_available = self.check_ollama_health()
        
        self.gemini_available = False
        try:
            import google.generativeai as genai
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai
                self.gemini_available = True
                print("✅ Gemini API initialized successfully")
            else:
                print("⚠️ GEMINI_API_KEY not found in environment variables")
                print("   Set it with: export GEMINI_API_KEY='your-api-key' (Linux/Mac)")
                print("   Or: $env:GEMINI_API_KEY='your-api-key' (Windows PowerShell)")
        except ImportError:
            print("⚠️ google-generativeai not installed. Install with: pip install google-generativeai")
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")

    def count_tokens(self, inputs):
        if isinstance(inputs, str):
            return len(inputs.split())
        elif isinstance(inputs, list):
            return [len(input_text.split()) for input_text in inputs]
        else:
            raise ValueError("Input must be a string or a list of strings.")

    @contextmanager
    def timer(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        yield
        if self.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        self.elapsed = end_time - start_time
    
    def generate_response(self, prompt, llm="tinyllama", max_tokens=512):
        llm_info = self.llm_map.get(llm, self.llm_map["tinyllama"])
        
        if llm_info["type"] == "ollama":
            try:
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": llm_info["model"],
                        "prompt": prompt,
                        "options": {"num_predict": max_tokens},
                        "stream": False
                    },
                    timeout=60
                )
                return response.json().get("response", "")
            except Exception as e:
                print(f"Ollama generation failed: {e}")
                return f"Error: {str(e)}"
        elif llm_info["type"] == "gemini":
            if not self.gemini_available:
                return "Error: Gemini API not available. Please set GEMINI_API_KEY environment variable."
            try:
                model = self.gemini_client.GenerativeModel(llm_info["model"])
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            except Exception as e:
                print(f"Gemini generation failed: {e}")
                return f"Error: {str(e)}"
        else:
            raise ValueError(f"Unknown LLM backend: {llm}")

    def generate_responses_batch(self, prompts, llm="tinyllama", max_tokens=512):
        results = []
        
        for prompt in prompts:
            try:
                result = self.generate_response(prompt, llm=llm, max_tokens=max_tokens)
                results.append(result)
            except Exception as e:
                print(f"Batch response generation failed for prompt: {e}")
                results.append(f"Error: {str(e)}")
        
        return results
    
    def get_embedding(self, text: str) -> torch.Tensor:
        cache_key = hash(text)
        if cache_key not in self._embedding_cache:
            try:
                with torch.no_grad():
                    embedding = self.semantic_model.encode(
                        text,
                        convert_to_tensor=True,
                        device=self.device,
                        show_progress_bar=False
                    )
                    self._embedding_cache[cache_key] = embedding
            except Exception as e:
                print(f"Embedding generation failed: {e}")
                self._embedding_cache[cache_key] = torch.zeros(384, device=self.device)
        
        return self._embedding_cache[cache_key]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            embedding1 = self.get_embedding(text1)
            embedding2 = self.get_embedding(text2)
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            return float(similarity)
        except Exception as e:
            print(f"Similarity calculation failed: {e}")
            return 0.8
    
    def get_llm_response(self, prompt: str, llm="tinyllama", max_tokens: int = 500) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            response_text = self.generate_response(prompt, llm=llm, max_tokens=max_tokens)
            end_time = time.time()
            
            result = {
                "response": response_text.strip(),
                "time_taken": end_time - start_time,
                "prompt_tokens": self.count_tokens(prompt),
                "completion_tokens": self.count_tokens(response_text),
                "total_tokens": self.count_tokens(prompt) + self.count_tokens(response_text),
                "success": True
            }
        except Exception as e:
            end_time = time.time()
            result = {
                "response": f"Error: {str(e)}",
                "time_taken": end_time - start_time,
                "prompt_tokens": self.count_tokens(prompt),
                "completion_tokens": 0,
                "total_tokens": self.count_tokens(prompt),
                "success": False
            }
        
        return result
    
    def optimize_tokens(self, prompt, target_reduction=0.3, min_similarity=0.85, llm="phi", category=None):
        original_tokens = self.count_tokens(prompt)
        import re
        if category is None:
            category = "generic"
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ['math', 'equation', 'calculate', 'derivative', 'integral', 'limit', 'solve', 'algebra', 'geometry', 'calculus', 'theorem', 'proof', 'matrix', 'vector', 'graph', 'plot', 'function', 'variable', 'constant']):
                category = "math"
            elif any(word in prompt_lower for word in ['code', 'function', 'algorithm', 'program', 'script', 'class', 'method', 'variable', 'loop', 'conditional', 'debug', 'compile', 'syntax', 'library', 'module', 'api', 'framework']):
                category = "coding"
            elif re.search(r'\b\d+\s*[\+\-\*/\^=<>]+\s*\d+\b', prompt_lower) or re.search(r'\b[a-zA-Z]\s*[\+\-\*/\^=<>]+\s*[a-zA-Z]\b', prompt_lower):
                category = "math"
            elif re.search(r'\b(def|class|import|for|if|while|try|except)\b', prompt_lower):
                category = "coding"
                
        
        strategies = []
        
        domain_optimizer = self._get_domain_optimizer(category)
        if domain_optimizer is not None:
            strategies.append(('domain_specific', domain_optimizer))
        
        strategies.extend([
            ('nlp_smart', lambda x: self.nlp_optimizer.smart_summarization(x, target_reduction, min_similarity)),
            ('nlp_dependency', lambda x: self.nlp_optimizer.dependency_based_optimization(x, target_reduction, min_similarity)),
            ('conservative', lambda x: self.preprocessor.preprocess(x, high_similarity=True))
        ])
        
        best_result = None
        best_score = -1
        
        for strategy_name, optimizer_func in strategies:
            try:
                optimized_prompt = optimizer_func(prompt)
                
                optimized_tokens = self.count_tokens(optimized_prompt)
                reduction = (original_tokens - optimized_tokens) / original_tokens if original_tokens > 0 else 0.0
                reduction_percent = reduction * 100
                similarity = self.calculate_similarity(prompt, optimized_prompt)
                
                meets_reduction = reduction_percent >= 30
                meets_similarity = similarity >= min_similarity
                
                if meets_reduction and meets_similarity:
                    score = 1000 + similarity + (reduction_percent / 100)
                elif meets_reduction:
                    score = 500 + similarity
                elif meets_similarity:
                    score = 100 + (reduction_percent / 100)
                else:
                    score = similarity + (reduction_percent / 100)
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'optimized_prompt': optimized_prompt,
                        'reduction_percent': reduction_percent,
                        'similarity': similarity,
                        'strategy': strategy_name,
                        'meets_criteria': meets_reduction and meets_similarity
                    }
                    
            except Exception as e:
                print(f"Strategy {strategy_name} failed: {e}")
                continue
        
        if not best_result:
            best_result = {
                'optimized_prompt': prompt,
                'reduction_percent': 0.0,
                'similarity': 1.0,
                'strategy': 'fallback',
                'meets_criteria': False
            }
        else:
            if best_result['similarity'] < min_similarity:
                print(f"Warning: Similarity too low ({best_result['similarity']:.3f}). Trying more conservative approach.")
                
                optimized_prompt = self.nlp_optimizer.smart_summarization(
                    prompt, 
                    target_reduction=target_reduction/1.5,
                    min_similarity=min_similarity+0.05
                )
                
                optimized_tokens = self.count_tokens(optimized_prompt)
                reduction_percent = ((original_tokens - optimized_tokens) / original_tokens * 100) if original_tokens else 0
                optimized_similarity = self.calculate_similarity(prompt, optimized_prompt)
                
                if optimized_similarity > best_result['similarity']:
                    best_result = {
                        'optimized_prompt': optimized_prompt,
                        'reduction_percent': reduction_percent,
                        'similarity': optimized_similarity,
                        'strategy': 'safety_fallback',
                        'meets_criteria': (reduction_percent >= 30) and (optimized_similarity >= min_similarity)
                    }
        
        target_achieved = (best_result['reduction_percent'] >= 30) and (best_result['similarity'] >= min_similarity)
        optimized_tokens = self.count_tokens(best_result['optimized_prompt'])
        tokens_saved = original_tokens - optimized_tokens
        
        metrics = {
            "token_reduction": best_result['reduction_percent'],
            "reduction_percent": best_result['reduction_percent'],
            "semantic_similarity": best_result['similarity'],
            "similarity": best_result['similarity'],
            "target_achieved": target_achieved,
            "strategy_used": best_result.get('strategy', 'unknown'),
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved
        }
        
        return best_result['optimized_prompt'], metrics
    
    def _get_domain_optimizer(self, category):
        if not self.nlp_optimizer.available:
            return None
            
        if category == "math":
            return lambda x: self._nlp_optimize_math(x)
        elif category == "coding":
            return lambda x: self._nlp_optimize_coding(x)
        elif category == "generic":
            return lambda x: self._nlp_optimize_generic(x)
        else:
            return lambda x: self._nlp_optimize_generic(x)
    
    def _nlp_optimize_math(self, prompt):
        try:
            if not self.nlp_optimizer.available:
                return self._optimize_math_prompt(prompt)
            
            doc = self.nlp_optimizer.nlp(prompt)
            important_tokens = []
            
            for token in doc:
                keep = False
                
                if token.like_num or token.text in '+-×÷=<>()[]{}√∫∂∆∑∏':
                    keep = True
                
                math_terms = {'function', 'equation', 'derivative', 'integral', 'limit', 
                             'sum', 'product', 'matrix', 'vector', 'graph', 'plot',
                             'x', 'y', 'z', 'n', 'i', 'j', 'k', 'f', 'g', 'h'}
                if token.text.lower() in math_terms:
                    keep = True
                
                if token.dep_ in ['ROOT', 'nsubj', 'dobj', 'pobj']:
                    keep = True
                
                if token.ent_type_:
                    keep = True
                
                if keep:
                    important_tokens.append(token.text)
            
            result = ' '.join(important_tokens)
            
            math_compressions = {
                'calculate the': 'find',
                'determine the': 'find', 
                'compute the': 'find',
                'find the value of': 'find',
                'what is the': 'find'
            }
            
            for verbose, short in math_compressions.items():
                result = result.replace(verbose, short)
            
            similarity = self.nlp_optimizer._calculate_similarity_gpu(prompt, result)
            if similarity < 0.85:
                result = self.nlp_optimizer.smart_summarization(prompt, 0.25, 0.85)
        
            return result if result.strip() else prompt
            
        except Exception as e:
            print(f"Math optimization failed: {e}")
            return self._optimize_math_prompt(prompt)

    def _nlp_optimize_coding(self, prompt):
        try:
            return self._optimize_coding_prompt(prompt)
        except Exception as e:
            print(f"Coding optimization failed: {e}")
            return prompt

    def _nlp_optimize_generic(self, prompt):
        try:
            if not self.nlp_optimizer.available:
                return prompt
            
            result = self.nlp_optimizer.smart_summarization(prompt, 0.35, 0.80)
            
            original_len = len(prompt.split())
            result_len = len(result.split())
            if original_len > 0:
                reduction = (original_len - result_len) / original_len
            else:
                reduction = 0.0
            
            if reduction < 0.25:
                print(f"[LLM_OPT] Low reduction ({reduction:.3f}) for prompt length {original_len}; trying dependency-based optimization.")
                result = self.nlp_optimizer.dependency_based_optimization(prompt, 0.35, 0.80)
            
            return result
            
        except Exception as e:
            print(f"Generic optimization failed: {e}")
            return prompt
    
    def _optimize_math_prompt(self, prompt):
        import re
        result = prompt
        
        replacements = {
            'calculate the': 'find',
            'determine the': 'find',
            'compute the': 'find',
            'find the value of': 'find',
            'what is the': 'find'
        }
        
        for k, v in replacements.items():
            result = result.replace(k, v)
        
        result = re.sub(r'\b(the|a|an|where|that|which|as|of|for|to|is|has|and|in|on|with|by|from|at)\b', '', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result if result.strip() else prompt

    def _optimize_coding_prompt(self, prompt):
        import re
        result = prompt
        
        replacements = {
            'Write a Python function that takes': 'Python func:',
            'Implement': 'Code:',
            'Create': 'Make:',
            'function': 'func'
        }
        
        for k, v in replacements.items():
            result = result.replace(k, v)
        
        result = re.sub(r'\b(the|a|an|that|which|as|of|for|to|is|has|and|in|on|with|by|from|at)\b', '', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result if result.strip() else prompt
    
    def run_efficiency_test(self, prompts: List[str], preprocess: bool = True, llm: str = "tinyllama") -> Dict[str, float]:
        print(f"🚀 Starting efficiency test on {self.device}")
        
        with self.timer():
            results = self.test_prompts(prompts, preprocess, llm=llm)
        
        test_time = self.elapsed
        summary = self.visualize_results(results)
        
        print("\n====== SUMMARY ======")
        print(f"Total test time: {test_time:.2f} seconds")
        print(f"Average token reduction: {summary['avg_token_reduction']:.2f}%")
        print(f"Average response similarity: {summary['avg_response_similarity']:.4f} ({summary['avg_response_similarity']*100:.2f}%)")
        
        if self.device == "cuda":
            allocated, reserved = GPUManager.get_memory_usage()
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        self._cleanup()
        return summary
    
    def test_prompts(self, prompts: List[str], preprocess: bool = True, llm: str = "tinyllama") -> List[Dict[str, Any]]:
        results = []
        batch_size = 8 if self.device == "cuda" else 2

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_results = []

            if preprocess:
                processed_batch = []
                for prompt in batch:
                    processed_batch.append(self.preprocess_prompt(prompt))
            else:
                processed_batch = batch


            for j, (prompt, processed_prompt) in enumerate(zip(batch, processed_batch)):
                test_id = i + j + 1
                print(f"\nTesting prompt {test_id}")

                try:
                    original_result = self.get_llm_response(prompt, llm=llm)
                    processed_result = self.get_llm_response(processed_prompt, llm=llm)

                    response_similarity = self.calculate_similarity(
                        original_result["response"], 
                        processed_result["response"]
                    )

                    token_reduction = original_result["prompt_tokens"] - processed_result["prompt_tokens"]
                    token_reduction_percent = (token_reduction / original_result["prompt_tokens"] * 100 
                                              if original_result["prompt_tokens"] > 0 else 0)

                    result = {
                        "test_id": test_id,
                        "original_prompt": prompt,
                        "processed_prompt": processed_prompt,
                        "original_tokens": original_result["prompt_tokens"],
                        "processed_tokens": processed_result["prompt_tokens"],
                        "token_reduction": token_reduction,
                        "token_reduction_percent": token_reduction_percent,
                        "response_similarity": response_similarity,
                        "original_response": original_result["response"],
                        "processed_response": processed_result["response"]
                    }

                    batch_results.append(result)

                except Exception as e:
                    print(f"Error processing prompt {test_id}: {e}")
                    continue

        results.extend(batch_results)

        if self.device == "cuda":
            GPUManager.clear_cache()

        return results
    
    def preprocess_prompt(self, prompt: str) -> str:
        if prompt not in self._preprocessed_cache:
            self._preprocessed_cache[prompt] = self.preprocessor.preprocess(prompt)
        return self._preprocessed_cache[prompt]
    
    def visualize_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"avg_token_reduction": 0.0, "avg_response_similarity": 0.0}
        
        df = pd.DataFrame(results)
        
        avg_token_reduction = df["token_reduction_percent"].mean()
        avg_response_similarity = df["response_similarity"].mean()
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(results))
        width = 0.35
        
        plt.bar(x - width/2, df["token_reduction_percent"], width, label='Token Reduction %', alpha=0.8)
        plt.bar(x + width/2, df["response_similarity"] * 100, width, label='Response Similarity %', alpha=0.8)
        
        plt.ylabel('Percentage')
        plt.title('GPU-Accelerated Token Reduction vs Response Similarity')
        plt.xticks(x, [f"Test {i+1}" for i in range(len(results))])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_folder}/efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        with open(f"{self.output_folder}/detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(f"{self.output_folder}/summary.txt", "w") as f:
            f.write("# GPU-ACCELERATED LLM PREPROCESSING EFFICIENCY TEST\n\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Average token reduction: {avg_token_reduction:.2f}%\n")
            f.write(f"Average response similarity: {avg_response_similarity:.4f} ({avg_response_similarity*100:.2f}%)\n\n")
        
        print(f"Results saved to {self.output_folder}")
        
        return {
            "avg_token_reduction": avg_token_reduction,
            "avg_response_similarity": avg_response_similarity
        }
    
    def _cleanup(self):
        print("🧹 Cleaning up resources...")
        
        self._preprocessed_cache.clear()
        self._embedding_cache.clear()
        
        GPUManager.clear_cache()
        
        gc.collect()
        
        if self.device == "cuda":
            allocated, reserved = GPUManager.get_memory_usage()
            print(f"🧹 GPU cleanup complete. Memory: {allocated:.2f} GB allocated")
    
    def _iterative_refinement(self, prompt, original_tokens, target_reduction, min_similarity, category=None):
        current_prompt = prompt
        
        if self.nlp_optimizer.available:
            if category == "math":
                current_prompt = self._nlp_optimize_math(current_prompt)
            elif category == "generic":
                current_prompt = self._nlp_optimize_generic(current_prompt)
            elif category == "coding":
                current_prompt = self._nlp_optimize_coding(current_prompt)
            else:
                current_prompt = self.nlp_optimizer.smart_summarization(current_prompt, target_reduction, min_similarity)
        
        current_tokens = self.count_tokens(current_prompt)
        reduction = (original_tokens - current_tokens) / original_tokens if original_tokens > 0 else 0.0
        reduction_percent = reduction * 100
        similarity = self.calculate_similarity(prompt, current_prompt)
        
        if reduction_percent < 30 or similarity < min_similarity:
            if self.nlp_optimizer.available:
                current_prompt = self.nlp_optimizer.dependency_based_optimization(
                    prompt, target_reduction, min_similarity
                )
                
                current_tokens = self.count_tokens(current_prompt)
                reduction = (original_tokens - current_tokens) / original_tokens if original_tokens > 0 else 0.0
                reduction_percent = reduction * 100
                similarity = self.calculate_similarity(prompt, current_prompt)
        
        return {
            'optimized_prompt': current_prompt,
            'reduction_percent': reduction_percent,
            'similarity': similarity,
            'strategy': 'nlp_iterative_refinement',
            'meets_criteria': reduction_percent >= 30 and similarity >= min_similarity
        }
    
    def calculate_llm_similarity(self, original_response, optimized_response, llm="tinyllama"):
        try:
            base_similarity = self.calculate_similarity(original_response, optimized_response)
            
            if llm in ["codellama", "qwen-coder"]:
                code_similarity = self._calculate_code_similarity(original_response, optimized_response)
                return (base_similarity + code_similarity) / 2
            elif llm in ["deepseek-math", "qwen-math"]:
                math_similarity = self._calculate_math_similarity(original_response, optimized_response)
                return (base_similarity + math_similarity) / 2
            else:
                return base_similarity
                
        except Exception as e:
            print(f"LLM similarity calculation failed: {e}")
            return base_similarity

    def _calculate_code_similarity(self, text1, text2):
        import re
        
        code1 = re.findall(r'```.*?```', text1, re.DOTALL)
        code2 = re.findall(r'```.*?```', text2, re.DOTALL)
        
        if not code1 or not code2:
            return self.calculate_similarity(text1, text2)
        
        return self.calculate_similarity(' '.join(code1), ' '.join(code2))

    def _calculate_math_similarity(self, text1, text2):
        import re
        
        math1 = re.findall(r'[\d\+\-\*/\=\(\)\^\√∫∂∆∑∏]+', text1)
        math2 = re.findall(r'[\d\+\-\*/\=\(\)\^\√∫∂∆∑∏]+', text2)
        
        if not math1 or not math2:
            return self.calculate_similarity(text1, text2)
        
        return self.calculate_similarity(' '.join(math1), ' '.join(math2))

    def progressive_optimization(self, prompt, category=None, max_attempts=3):
        strategies = [
            (0.15, 0.85),
            (0.25, 0.82),
            (0.35, 0.80),
        ]
        
        for attempt, (reduction, similarity) in enumerate(strategies):
            try:
                optimized, metrics = self.optimize_tokens(
                    prompt, 
                    target_reduction=reduction, 
                    min_similarity=similarity, 
                    category=category
                )
                
                if metrics['target_achieved']:
                    print(f"✅ Success with strategy {attempt + 1}")
                    return optimized, metrics
                    
            except Exception as e:
                print(f"Strategy {attempt + 1} failed: {e}")
                continue
        
        return optimized, metrics

    def safe_generate_response(self, prompt, llm="tinyllama", max_tokens=512, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self.generate_response(prompt, llm=llm, max_tokens=max_tokens)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return f"Failed after {max_retries} attempts: {str(e)}"
    
    def check_ollama_health(self):
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is available")
                return True
            else:
                print("❌ Ollama responded but with error")
                return False
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            return False

    def check_model_availability(self, llm="tinyllama"):
        try:
            import requests
            import re
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                llm_info = self.llm_map.get(llm, {})
                target_model = llm_info.get("model", "")
                
                available = any(target_model in model_name for model_name in model_names)
                return available
            return False
        except Exception:
            return False

if __name__ == "__main__":
    test_prompts = [
        "Calculate the derivative of x^2 + 3x using calculus principles and explain the process step by step.",
        "Write a Python function that implements binary search algorithm with proper error handling.",
        "Analyze the quarterly sales performance and provide actionable recommendations for improvement."
    ]
    
    tester = LLMEfficiencyTest(output_folder="./test_results")
    tester.run_efficiency_test(test_prompts, llm="tinyllama")
