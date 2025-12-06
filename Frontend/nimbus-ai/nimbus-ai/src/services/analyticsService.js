// services/analyticsService.js
// Handles analytics-related API calls and data processing

import apiService from './api.js';

export class AnalyticsService {
  // Transform single API response to analytics format
  static transformApiResponseToAnalytics(apiResponse) {
  if (!apiResponse || !apiResponse.data) return null;

  const d = apiResponse.data;

  const promptEntry = {
    id: `prompt-${apiResponse.timestamp || Date.now()}`,
    type: d.category || 'generic',
    original_prompt: d.original_prompt,
    optimized_prompt: d.optimized_prompt,
    llm_response: d.response || null,
    response_error: d.response_error || null,
    strategy_used: d.strategy_used,
    selected_llm: d.selected_llm,
    target_achieved: d.target_achieved,
    processing_time: apiResponse.processing_time,
    metrics: {
  token_reduction_percent: d.token_reduction_percent || 0,
  similarity: d.similarity || 0,
  original_tokens: d.metrics?.original_tokens || 0,
  optimized_tokens: d.metrics?.optimized_tokens || 0,
  tokens_saved: d.metrics?.tokens_saved || 0,
},

    cost: d.cost || null,
  };

  const summary = {
  total_queries: 1,
  token_reduction_percent: d.token_reduction_percent || 0,
  similarity: d.similarity || 0,
  tokens: {
    original_tokens: d.metrics?.original_tokens || 0,
    optimized_tokens: d.metrics?.optimized_tokens || 0,
    tokens_saved: d.metrics?.tokens_saved || 0,
  },
  cost: d.cost || null,
  strategy_used: d.strategy_used,
  selected_llm: d.selected_llm,
  target_achieved: d.target_achieved,
};


  return {
    summary,
    prompts: [promptEntry],
    raw_api_response: apiResponse,
  };
}


  // Get min similarity threshold based on strategy
  static getMinSimilarityForStrategy(strategy) {
    const thresholds = {
      aggressive: 0.85,
      balanced: 0.85,
      conservative: 0.90,
    };
    return thresholds[strategy] || 0.85;
  }

  // Get target reduction based on strategy
  static getTargetReductionForStrategy(strategy) {
    const targets = {
      aggressive: 35,
      balanced: 30,
      conservative: 15,
    };
    return targets[strategy] || 30;
  }

  // Calculate costs based on token counts
  // Using approximate pricing: $0.01 per 1000 tokens
  static calculateCosts(originalTokens, optimizedTokens) {
    const pricePerThousandTokens = 0.01;
    
    const standardModelCost = (originalTokens / 1000) * pricePerThousandTokens;
    const optimizedModelCost = (optimizedTokens / 1000) * pricePerThousandTokens;
    
    return {
      standard_model_cost: parseFloat(standardModelCost.toFixed(4)),
      optimized_model_cost: parseFloat(optimizedModelCost.toFixed(4)),
    };
  }

  // Process cost data for chart visualization
  static prepareCostData(summary) {
  if (!summary || !summary.cost) return [];

  return [
    {
      name: 'Original',
      cost: summary.cost.original_cost_usd,
    },
    {
      name: 'Optimized',
      cost: summary.cost.optimized_cost_usd,
    },
  ];
}


  // Calculate metrics for display
  static calculateMetrics(summary) {
  if (!summary || !summary.cost) {
    return { totalSavings: '0.000000', savingsPercentage: '0.0' };
  }

  const original = summary.cost.original_cost_usd || 0;
  const optimized = summary.cost.optimized_cost_usd || 0;

  const savings = original - optimized;
  const savingPercent =
    original > 0 ? ((savings / original) * 100).toFixed(1) : '0.0';

  return {
    totalSavings: savings.toFixed(6),
    savingsPercentage: savingPercent,
  };
}

  // Get available strategies from API
  static async getStrategies() {
    try {
      const response = await apiService.get('/strategies');
      return { success: true, data: response };
    } catch (error) {
      console.error('Failed to fetch strategies:', error);
      return { success: false, error: error.message };
    }
  }

  // Health check
  static async checkHealth() {
    try {
      const response = await apiService.get('/health');
      return { success: true, data: response };
    } catch (error) {
      console.error('Health check failed:', error);
      return { success: false, error: error.message };
    }
  }

  // Get API status
  static async getStatus() {
    try {
      const response = await apiService.get('/status');
      return { success: true, data: response };
    } catch (error) {
      console.error('Status check failed:', error);
      return { success: false, error: error.message };
    }
  }
}