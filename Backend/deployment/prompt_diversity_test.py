
from llm_efficiency_test import LLMEfficiencyTest
import json
import os
from datetime import datetime
import concurrent.futures
import csv
import time
import traceback
import torch
import pandas as pd
import numpy as np
import re



class PromptDiversityTester:

    def __init__(self, llm_efficiency_test=None, prompts_dir="./prompts"):
        if llm_efficiency_test is None:
            from llm_efficiency_test import LLMEfficiencyTest
            self.llm_efficiency_test = LLMEfficiencyTest()
        else:
            self.llm_efficiency_test = llm_efficiency_test
        
        self.prompts_dir = prompts_dir
        self.test_prompts = self.load_prompts_from_csv()
        
        self.prompt_files = {
            "coding": "./prompts/coding.csv",
            "math": "./prompts/math.csv", 
            "generic": "./prompts/generic.csv"
        }

    def load_prompts_from_csv(self):
        test_prompts = {}
        if not os.path.isdir(self.prompts_dir):
            raise FileNotFoundError(f"Prompts directory '{self.prompts_dir}' not found.")
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".csv"):
                category = filename.replace(".csv", "")
                filepath = os.path.join(self.prompts_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    prompts = [row[0] for row in reader if row]
                    if prompts:
                        test_prompts[category] = prompts
        if not test_prompts:
            raise ValueError("No prompts loaded from CSV files.")
        return test_prompts
    
    def load_prompts(self, category):
        try:
            file_path = self.prompt_files[category]
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return []
            
            df = pd.read_csv(file_path)
            
            if 'prompt' in df.columns:
                prompts = df['prompt'].dropna().tolist()
            elif 'Prompt' in df.columns:
                prompts = df['Prompt'].dropna().tolist()
            elif 'text' in df.columns:
                prompts = df['text'].dropna().tolist()
            else:
                prompts = df.iloc[:, 0].dropna().tolist()
            
            prompts = prompts[:200]
            
            print(f"📁 Loaded {len(prompts)} {category} prompts from {file_path}")
            return prompts
            
        except Exception as e:
            print(f"❌ Error loading {category} prompts: {e}")
            return []

    def test_all_categories(self, target_reduction=0.30, min_similarity=0.80, llm="tinyllama", batch_size=10):
        results = {}
        overall_stats = {
            'total_prompts': 0,
            'successful_optimizations': 0,
            'avg_reduction': 0,
            'avg_similarity': 0
        }

        def process_batch(batch):
            batch_results = []
            for args in batch:
                result = process_prompt(args)
                batch_results.append(result)
                overall_stats['total_prompts'] += 1
                if result['rl_ready']:
                    overall_stats['successful_optimizations'] += 1
                overall_stats['avg_reduction'] += result['token_reduction']
                overall_stats['avg_similarity'] += result['semantic_similarity']
            return batch_results

        def process_prompt(args):
            category, i, prompt = args
            try:
                print(f"\nTesting {category} prompt {i}...")
                optimized, metrics = self.llm_efficiency_test.optimize_tokens(
                    prompt, target_reduction=target_reduction, min_similarity=min_similarity, llm=llm, category=category
                )
                llm_similarity = 0
                if metrics['target_achieved']:
                    try:
                        original_response = self.llm_efficiency_test.get_llm_response(prompt, llm=llm)
                        optimized_response = self.llm_efficiency_test.get_llm_response(optimized, llm=llm)
                        llm_similarity = self.llm_efficiency_test.calculate_similarity(
                            original_response['response'], optimized_response['response']
                        )
                    except Exception as e:
                        print(f"LLM testing failed: {e}")
                        llm_similarity = metrics['similarity']
                else:
                    llm_similarity = metrics['similarity']
                result = {
                    'original_prompt': prompt,
                    'optimized_prompt': optimized,
                    'token_reduction': metrics['reduction_percent'],
                    'semantic_similarity': metrics['similarity'],
                    'llm_response_similarity': llm_similarity,
                    'target_achieved': metrics['target_achieved'],
                    'rl_ready': (metrics['target_achieved'] and 
                                metrics['reduction_percent'] >= 30 and 
                                metrics['similarity'] >= 0.8 and 
                                llm_similarity >= 0.80),
                    'llm_used': llm
                }
                print(f"  Reduction: {result['token_reduction']:.1f}%")
                print(f"  Semantic Sim: {result['semantic_similarity']:.3f}")
                print(f"  LLM Sim: {result['llm_response_similarity']:.3f}")
                print(f"  RL Ready: {'✅' if result['rl_ready'] else '❌'}")
                return result
            except Exception as e:
                print(f"Exception in prompt {category} #{i}: {e}")
                traceback.print_exc()
                return {
                    'original_prompt': prompt,
                    'optimized_prompt': '',
                    'token_reduction': 0.0,
                    'semantic_similarity': 0.0,
                    'llm_response_similarity': 0.0,
                    'target_achieved': False,
                    'rl_ready': False,
                    'llm_used': llm
                }

        for category, prompts in self.test_prompts.items():
            print(f"\n{'='*60}")
            print(f"TESTING {category.upper()} PROMPTS")
            print(f"{'='*60}")

            category_results = []
            args_list = [(category, i, prompt) for i, prompt in enumerate(prompts, 1)]

            for i in range(0, len(args_list), batch_size):
                batch = args_list[i:i + batch_size]
                batch_results = process_batch(batch)
                category_results.extend(batch_results)

            results[category] = category_results

        total_prompts = overall_stats['total_prompts']
        overall_stats['avg_reduction'] /= total_prompts
        overall_stats['avg_similarity'] /= total_prompts
        overall_stats['success_rate'] = (overall_stats['successful_optimizations'] / total_prompts) * 100

        return results, overall_stats
    
    def classify_prompt(self, prompt):
        
        prompt_lower = prompt.lower()
        
        coding_patterns = [
            r'\bdef\s+\w+',
            r'\bimport\s+\w+',
            r'\bclass\s+\w+',
            r'\bfor\s+\w+\s+in\b',
            r'\bif\s+.*:',
            r'\bprint\s*\(',
            r'\breturn\s+',
            r'\bvariable\b',
            r'\bcode\b',
            r'\bfunction\b',
            r'\balgorithm\b',
            r'\bscript\b',
            r'\bprogramming\b'
        ]
        
        math_patterns = [
            r'\d+[\+\-\*\/=]\d+',
            r'\bequation\b',
            r'\bsolve\s+(for|the)?',
            r'\bcalculate\b',
            r'\bformula\b',
            r'\btheorem\b',
            r'\bproof\b',
            r'\bintegral\b',
            r'\bderivative\b',
            r'\bmatrix\b',
            r'\bvector\b',
            r'\bprobability\b',
            r'\bstatistics\b'
        ]
        
        image_gen_patterns = [
            r'\bgenerate\s+image\b',
            r'\bcreate\s+image\b',
            r'\bdraw\s+',
            r'\bpaint\s+',
            r'\bart\b',
            r'\bdesign\b',
            r'\billustrate\b',
            r'\bvisualize\b'
        ]
        
        image_input_patterns = [
            r'\bclassify\s+image\b',
            r'\banalyze\s+image\b',
            r'\bdescribe\s+image\b',
            r'\bdetect\s+in\s+image\b',
            r'\brecognize\s+',
            r'\bidentify\s+',
            r'\bsegment\s+',
            r'\bocr\b',
            r'\bcomputer\s+vision\b'
        ]
        
        if any(re.search(pattern, prompt, re.IGNORECASE) for pattern in coding_patterns):
            return "coding"
        elif any(re.search(pattern, prompt, re.IGNORECASE) for pattern in math_patterns):
            return "math"
        elif any(re.search(pattern, prompt, re.IGNORECASE) for pattern in image_gen_patterns):
            return "image_generation"
        elif any(re.search(pattern, prompt, re.IGNORECASE) for pattern in image_input_patterns):
            return "image_input"
        else:
            return "generic"

    def route_prompt_to_llm(self, prompt, prefer_gemini=False):
        category = self.classify_prompt(prompt)
        
        gemini_available = hasattr(self.llm_efficiency_test, 'gemini_available') and \
                          self.llm_efficiency_test.gemini_available
        
        if prefer_gemini and gemini_available:
            llm_mapping = {
                "coding": "gemini-pro",
                "math": "gemini-pro",
                "generic": "gemini-flash",
                "image_input": "llava",
                "image_generation": "moondream"
            }
        else:
            llm_mapping = {
                "coding": "codellama",
                "math": "qwen2-math",
                "generic": "tinyllama",
                "image_input": "llava",
                "image_generation": "moondream"
            }
        
        llm = llm_mapping.get(category, "tinyllama")
        print(f"🛠 Routing prompt to LLM: {llm} (Category: {category})")
        return llm

    def process_prompt(self, prompt, target_reduction=0.3, min_similarity=0.8, max_length=512):
        try:
            llm = self.route_prompt_to_llm(prompt)
            category = self.classify_prompt(prompt)

            optimized_prompt, metrics = self.llm_efficiency_test.optimize_tokens(
                prompt, 
                target_reduction=target_reduction, 
                min_similarity=min_similarity, 
                llm=llm, 
                category=category,
                max_length=max_length
            )
            
            original_tokens = self.llm_efficiency_test.count_tokens(prompt)
            optimized_tokens = self.llm_efficiency_test.count_tokens(optimized_prompt)
            
            llm_similarity = 0.0
            try:
                if metrics.get('target_achieved', False):
                    original_response = self.llm_efficiency_test.get_llm_response(prompt, llm=llm, max_length=max_length)
                    optimized_response = self.llm_efficiency_test.get_llm_response(optimized_prompt, llm=llm, max_length=max_length)
                    llm_similarity = self.llm_efficiency_test.calculate_similarity(
                        original_response['response'], optimized_response['response']
                    )
                else:
                    llm_similarity = metrics.get('similarity', 0.0)
            except Exception as e:
                print(f"⚠️  LLM response similarity test failed: {e}")
                llm_similarity = metrics.get('similarity', 0.0)
            
            result = {
                'category': category,
                'original_prompt': prompt,
                'optimized_prompt': optimized_prompt,
                'original_tokens': original_tokens,
                'optimized_tokens': optimized_tokens,
                'reduction_percent': metrics.get('reduction_percent', 0),
                'similarity': metrics.get('similarity', 0),
                'llm_similarity': llm_similarity,
                'target_achieved': metrics.get('target_achieved', False),
                'strategy_used': metrics.get('strategy_used', 'unknown'),
                'llm_used': llm,
                'rl_ready': (
                    metrics.get('target_achieved', False) and 
                    metrics.get('reduction_percent', 0) >= 30 and 
                    metrics.get('similarity', 0) >= 0.8 and 
                    llm_similarity >= 0.80
                ),
                'state': {
                    'original_prompt': prompt,
                    'category': category,
                    'original_tokens': original_tokens,
                    'llm_used': llm
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing prompt: {e}")
            return {
                'category': "unknown",
                'original_prompt': prompt,
                'optimized_prompt': prompt,
                'original_tokens': 0,
                'optimized_tokens': 0,
                'reduction_percent': 0,
                'similarity': 0,
                'llm_similarity': 0,
                'target_achieved': False,
                'strategy_used': 'failed',
                'llm_used': "unknown",
                'rl_ready': False,
                'state': {
                    'original_prompt': prompt,
                    'category': "unknown",
                    'original_tokens': 0,
                    'llm_used': "unknown"
                }
            }

    def analyze_results(self, results, overall_stats=None):
        if not results:
            return {
                'overall_stats': {
                    'total_prompts': 0,
                    'successful_optimizations': 0,
                    'avg_reduction': 0,
                    'avg_similarity': 0,
                    'success_rate': 0
                },
                'category_results': {},
                'rl_readiness': False
            }
        
        if overall_stats is not None:
            print(f"\n{'='*80}")
            print("RL READINESS ANALYSIS")
            print(f"{'='*80}")
            
            print(f"\nOVERALL PERFORMANCE:")
            print(f"Success Rate: {overall_stats['success_rate']:.1f}%")
            print(f"Average Reduction: {overall_stats['avg_reduction']:.1f}%")
            print(f"Average Similarity: {overall_stats['avg_similarity']:.3f}")
            print(f"RL Ready: {'✅ YES' if overall_stats['success_rate'] >= 80 else '❌ NEEDS IMPROVEMENT'}")
            
            if isinstance(results, dict):
                for category, category_results in results.items():
                    if category_results:
                        avg_reduction = sum(r['token_reduction'] for r in category_results) / len(category_results)
                        avg_similarity = sum(r['semantic_similarity'] for r in category_results) / len(category_results)
                        avg_llm_sim = sum(r['llm_response_similarity'] for r in category_results) / len(category_results)
                        rl_ready_count = sum(1 for r in category_results if r['rl_ready'])
                        rl_ready_rate = (rl_ready_count / len(category_results)) * 100
                        
                        print(f"\n{category.upper()}:")
                        print(f"  Average reduction: {avg_reduction:.1f}%")
                        print(f"  Average semantic similarity: {avg_similarity:.3f}")
                        print(f"  Average LLM similarity: {avg_llm_sim:.3f}")
                        print(f"  RL ready rate: {rl_ready_rate:.1f}%")
            
            return overall_stats['success_rate'] >= 80
        
        import numpy as np
        
        total_prompts = len(results)
        successful_optimizations = sum(1 for r in results if r.get('target_achieved', False))
        
        reductions = [r.get('reduction_percent', 0) for r in results if 'reduction_percent' in r]
        similarities = [r.get('similarity', 0) for r in results if 'similarity' in r]
        
        avg_reduction = np.mean(reductions) if reductions else 0
        avg_similarity = np.mean(similarities) if similarities else 0
        success_rate = (successful_optimizations / total_prompts) * 100 if total_prompts > 0 else 0
        
        category_results = {}
        for category in ['coding', 'math', 'generic']:
            category_data = [r for r in results if r.get('category') == category]
            
            if category_data:
                cat_successful = sum(1 for r in category_data if r.get('target_achieved', False))
                cat_total = len(category_data)
                cat_success_rate = (cat_successful / cat_total) * 100 if cat_total > 0 else 0
                
                cat_reductions = [r.get('reduction_percent', 0) for r in category_data]
                cat_similarities = [r.get('similarity', 0) for r in category_data]
                
                category_results[category] = {
                    'total': cat_total,
                    'successful': cat_successful,
                    'success_rate': cat_success_rate,
                    'avg_reduction': np.mean(cat_reductions) if cat_reductions else 0,
                    'avg_similarity': np.mean(cat_similarities) if cat_similarities else 0,
                    'results': category_data
                }
            else:
                category_results[category] = {
                    'total': 0,
                    'successful': 0,
                    'success_rate': 0,
                    'avg_reduction': 0,
                    'avg_similarity': 0,
                    'results': []
                }
        
        rl_ready = success_rate >= 80.0
        
        analysis = {
            'overall_stats': {
                'total_prompts': total_prompts,
                'successful_optimizations': successful_optimizations,
                'avg_reduction': avg_reduction,
                'avg_similarity': avg_similarity,
                'success_rate': success_rate
            },
            'category_results': category_results,
            'rl_readiness': rl_ready
        };
        
        return analysis
    
    def save_results(self, results, overall_stats):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        full_data = {
            'timestamp': timestamp,
            'overall_stats': overall_stats,
            'category_results': results,
            'rl_readiness': overall_stats['success_rate'] >= 80
        }
        
        json_filename = f"{results_dir}/rl_readiness_analysis_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)
        
        rl_training_data = []
        for category, category_results in results.items():
            for result in category_results:
                if result['rl_ready']:
                    rl_training_data.append({
                        'state': {
                            'original_prompt': result['original_prompt'],
                            'category': category,
                            'original_tokens': len(result['original_prompt'].split())
                        },
                        'action': {
                            'optimization_applied': 'quality_first',
                            'target_reduction': 0.3,
                            'min_similarity': 0.80
                        },
                        'reward': {
                            'token_reduction': result['token_reduction'] / 100,
                            'semantic_similarity': result['semantic_similarity'],
                            'llm_similarity': result['llm_response_similarity'],
                            'composite_reward': (result['token_reduction']/100 * 0.4 + 
                                               result['semantic_similarity'] * 0.3 + 
                                               result['llm_response_similarity'] * 0.3)
                        },
                        'next_state': {
                            'optimized_prompt': result['optimized_prompt'],
                            'optimized_tokens': len(result['optimized_prompt'].split())
                        }
                    })
        
        rl_filename = f"{results_dir}/rl_training_data_{timestamp}.json"
        with open(rl_filename, 'w', encoding='utf-8') as f:
            json.dump(rl_training_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved:")
        print(f"   - Analysis: {json_filename}")
        print(f"   - RL Data: {rl_filename}")
        
        return json_filename, rl_filename

    def generate_report(self, analysis_results, llm):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"📊 PROMPT DIVERSITY TEST RESULTS")
        print(f"{'='*60}")
        print(f"LLM: {llm}")
        print(f"Timestamp: {timestamp}")
        print(f"Total Prompts: {analysis_results['overall_stats']['total_prompts']}")
        print(f"Successful Optimizations: {analysis_results['overall_stats']['successful_optimizations']}")
        print(f"Average Reduction: {analysis_results['overall_stats']['avg_reduction']:.1f}%")
        print(f"Average Similarity: {analysis_results['overall_stats']['avg_similarity']:.3f}")
        print(f"Success Rate: {analysis_results['overall_stats']['success_rate']:.1f}%")
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        print(f"{'-'*60}")
        
        for category, data in analysis_results['category_results'].items():
            if data['total'] > 0:
                print(f"{category.upper()}:")
                print(f"  Average reduction: {data['avg_reduction']:.1f}%")
                print(f"  Average similarity: {data['avg_similarity']:.3f}")
                print(f"  RL ready rate: {data['success_rate']:.1f}%")
                print()
        
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        report_file = f"{results_dir}/rl_readiness_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'overall_stats': analysis_results['overall_stats'],
                'category_results': {k: {
                    'total': v['total'],
                    'successful': v['successful'], 
                    'success_rate': v['success_rate'],
                    'avg_reduction': v['avg_reduction'],
                    'avg_similarity': v['avg_similarity']
                } for k, v in analysis_results['category_results'].items()},
                'rl_readiness': analysis_results['rl_readiness']
            }, f, indent=2)
        
        if analysis_results['rl_readiness']:
            training_data = []
            for category_data in analysis_results['category_results'].values():
                for result in category_data['results']:
                    if result.get('target_achieved', False):
                        training_data.append({
                            "state": {
                                'category': result.get('category'),
                                'original_prompt': result.get('original_prompt'),
                                'optimized_prompt': result.get('optimized_prompt'),
                                'reduction_percent': result.get('reduction_percent'),
                                'similarity': result.get('similarity'),
                                'strategy': result.get('strategy_used')
                            }
                        })

            training_file = f"{results_dir}/rl_training_data_{timestamp}.json"
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            print(f"💾 Training data saved: {training_file}")
        
        print(f"💾 Analysis saved: {report_file}")
        
        return analysis_results

def test_rl_readiness_multi_llm(llms=None, batch_size=None, max_samples_per_llm=100):
    
    if llms is None:
        llms = [
            "tinyllama",
            "phi",
            "qwen",
            "codellama",
            "qwen-coder",
            "deepseek-math",
            "qwen-math",
        ]
    
    print(f"🚀 Multi-LLM Training Data Collection")
    print(f"📊 Testing {len(llms)} models: {', '.join(llms)}")
    print(f"🎯 Max {max_samples_per_llm} samples per model")
    print(f"=" * 80)
    
    all_results = []
    llm_performance = {}
    combined_stats = {
        'total_prompts': 0,
        'total_successful': 0,  
        'llm_distribution': {},
        'category_distribution': {},
        'overall_success_rate': 0
    }
    
    category_llm_preferences = {
        "math": ["deepseek-math", "qwen-math", "phi", "tinyllama"],
        "coding": ["codellama", "qwen-coder", "phi", "tinyllama"],
        "generic": ["phi", "qwen", "tinyllama"]
    }
    
    for llm_idx, llm in enumerate(llms):
        try:
            print(f"\n🔄 [{llm_idx + 1}/{len(llms)}] Collecting data with {llm.upper()}")
            print(f"-" * 60)
            
            rl_ready, llm_results, llm_stats = test_rl_readiness(llm=llm, batch_size=batch_size)
            
            if len(llm_results) > max_samples_per_llm:
                category_samples = {}
                for result in llm_results:
                    category = result.get('category', 'unknown')
                    if category not in category_samples:
                        category_samples[category] = []
                    category_samples[category].append(result)
                
                sampled_results = []
                samples_per_category = max_samples_per_llm // len(category_samples)
                
                for category, samples in category_samples.items():
                    if len(samples) <= samples_per_category:
                        sampled_results.extend(samples)
                    else:
                        import random
                        sampled_results.extend(random.sample(samples, samples_per_category))
                
                llm_results = sampled_results[:max_samples_per_llm]
                print(f"📉 Reduced to {len(llm_results)} balanced samples")
            
            for result in llm_results:
                result['llm_used'] = llm
                result['llm_ready'] = rl_ready
                
                category = result.get('category', 'unknown')
                combined_stats['category_distribution'][category] = \
                    combined_stats['category_distribution'].get(category, 0) + 1
                combined_stats['llm_distribution'][llm] = \
                    combined_stats['llm_distribution'].get(llm, 0) + 1
            
            all_results.extend(llm_results)
            llm_performance[llm] = {
                'rl_ready': rl_ready,
                'samples_collected': len(llm_results),
                'success_rate': llm_stats['overall_stats']['success_rate'],
                'avg_reduction': llm_stats['overall_stats']['avg_reduction'],
                'avg_similarity': llm_stats['overall_stats']['avg_similarity']
            }
            
            combined_stats['total_prompts'] += len(llm_results)
            combined_stats['total_successful'] += sum(1 for r in llm_results if r.get('rl_ready', False))
            
            print(f"✅ {llm}: {len(llm_results)} samples, {llm_stats['overall_stats']['success_rate']:.1f}% success")
            
        except Exception as e:
            print(f"❌ Error with {llm}: {str(e)}")
            llm_performance[llm] = {'error': str(e), 'samples_collected': 0}
            continue
    
    if combined_stats['total_prompts'] > 0:
        combined_stats['overall_success_rate'] = \
            (combined_stats['total_successful'] / combined_stats['total_prompts']) * 100
    
    print(f"\n" + "="*80)
    print(f"📊 MULTI-LLM DATA COLLECTION SUMMARY")
    print(f"="*80)
    print(f"Total samples collected: {combined_stats['total_prompts']}")
    print(f"Overall success rate: {combined_stats['overall_success_rate']:.1f}%")
    print(f"\n📈 LLM Performance:")
    for llm, perf in llm_performance.items():
        if 'error' in perf:
            print(f"  ❌ {llm}: {perf['error']}")
        else:
            print(f"  ✅ {llm}: {perf['samples_collected']} samples, {perf['success_rate']:.1f}% success")
    
    print(f"\n📊 Distribution:")
    print(f"  LLMs: {dict(combined_stats['llm_distribution'])}")
    print(f"  Categories: {dict(combined_stats['category_distribution'])}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    
    training_data_file = f"./results/multi_llm_training_data_{timestamp}.json"
    os.makedirs("./results", exist_ok=True)
    with open(training_data_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    analysis_file = f"./results/multi_llm_analysis_{timestamp}.json"
    analysis_data = {
        'timestamp': timestamp,
        'llm_performance': llm_performance,
        'combined_stats': combined_stats,
        'collection_params': {
            'llms_tested': llms,
            'max_samples_per_llm': max_samples_per_llm,
            'batch_size': batch_size
        }
    }
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"\n💾 Saved:")
    print(f"  Training data: {training_data_file}")
    print(f"  Analysis: {analysis_file}")
    
    return all_results, combined_stats, llm_performance

def test_rl_readiness(llm="tinyllama", batch_size=None):
    
    if batch_size is None:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = min(32, max(4, int(gpu_memory // 2)))
        else:
            batch_size = 1
    
    print(f"🚀 Using batch size: {batch_size}")
    
    llm_efficiency_test = LLMEfficiencyTest()
    
    tester = PromptDiversityTester(llm_efficiency_test)
    
    target_reduction = 0.3
    min_similarity = 0.8
    results = []
    
    for category in ["coding", "math", "generic"]:
        print(f"\n{'='*50}")
        print(f"🔍 Testing {category.upper()} prompts...")
        print(f"{'='*50}")
        
        prompts = tester.load_prompts(category)
        category_results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = tester.process_prompt(prompt, category, llm, target_reduction, min_similarity)
                category_results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1}/{len(prompts)} {category} prompts")
                    
            except Exception as e:
                print(f"❌ Error in {category} prompt #{i + 1}: {e}")
                continue
        
        results.extend(category_results)
        print(f"📊 {category.upper()}: {len(category_results)} prompts processed")
    
    analysis_results = tester.analyze_results(results)
    
    report = tester.generate_report(analysis_results, llm)
    
    overall_success_rate = analysis_results['overall_stats']['success_rate']
    rl_ready = overall_success_rate >= 80.0
    
    print(f"\n{'='*60}")
    print(f"🎯 RL READINESS ASSESSMENT")
    print(f"{'='*60}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"RL Ready: {'✅ YES' if rl_ready else '❌ NO'}")
    print(f"{'='*60}")
    
    return rl_ready, results, analysis_results

if __name__ == "__main__":
    import sys
    import re

    print("🤖 RL Training Data Collection Options:")
    print("1. Single LLM test (original)")
    print("2. Multi-LLM comprehensive collection (NEW)")

    if len(sys.argv) > 2 and sys.argv[1] == "prompt":
        user_input = " ".join(sys.argv[2:])
        print(f"🔍 Received prompt: {user_input}")

        tester = PromptDiversityTester()
        result = tester.process_prompt(user_input)
        print("✅ Processed Result:", result)

    elif len(sys.argv) > 1 and sys.argv[1] == "multi":
        start_time = time.time()
        print("\n🚀 Starting Multi-LLM Data Collection...")
        all_results, combined_stats, llm_performance = test_rl_readiness_multi_llm(
            max_samples_per_llm=100
        )
        end_time = time.time()
        print(f"\n⏱️  Total time taken: {end_time - start_time:.2f} seconds")
        print(f"✅ Collected {len(all_results)} diverse training samples!")
    else:
        start_time = time.time()
        print("\n🔄 Running single LLM test with TinyLlama...")
        test_rl_readiness(llm="tinyllama")
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print("\n💡 Tip: Run 'python prompt_diversity_test.py multi' for diverse data collection")
