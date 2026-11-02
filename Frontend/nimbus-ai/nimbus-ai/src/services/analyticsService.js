// Build analytics from real saved chat responses; no sample values are injected.

import apiService from './api.js';

function toPromptEntry(message) {
  const apiResponse = message?.fullApiResponse;
  const data = apiResponse?.data;
  if (!data) return null;
  return {
    id: message.id,
    category: data.category || 'generic',
    type: data.category || 'generic',
    original_prompt: data.original_prompt || message.originalPrompt || '',
    optimized_prompt: data.optimized_prompt || message.optimizedPrompt || '',
    llm_response: data.response || null,
    response_error: data.response_error || null,
    strategy_used: data.strategy_used,
    selected_llm: data.selected_llm,
    target_achieved: data.target_achieved,
    processing_time: Number(apiResponse.processing_time || 0),
    cache_hit: Boolean(data.cache_hit),
    feedback: message.feedback || null,
    metrics: {
      token_reduction_percent: Number(data.token_reduction_percent || 0),
      similarity: Number(data.similarity || 0),
      original_tokens: Number(data.metrics?.original_tokens || 0),
      optimized_tokens: Number(data.metrics?.optimized_tokens || 0),
      tokens_saved: Number(data.metrics?.tokens_saved || 0),
      token_count_method: data.metrics?.token_count_method,
    },
    cost: data.cost || null,
  };
}

export class AnalyticsService {
  static transformMessagesToAnalytics(messages, selectedMessage) {
    const entries = (messages || [])
      .filter((message) => message.type === 'assistant' && !message.isError && message.fullApiResponse)
      .map(toPromptEntry)
      .filter(Boolean);
    if (!entries.length) return null;

    const originalTokens = entries.reduce((sum, item) => sum + item.metrics.original_tokens, 0);
    const optimizedTokens = entries.reduce((sum, item) => sum + item.metrics.optimized_tokens, 0);
    const tokensSaved = entries.reduce((sum, item) => sum + item.metrics.tokens_saved, 0);
    const similarity = entries.reduce((sum, item) => sum + item.metrics.similarity, 0) / entries.length;
    const priced = entries.filter((item) => item.cost && Number.isFinite(Number(item.cost.optimized_cost_usd)));
    const cost = priced.length ? {
      original_cost_usd: priced.reduce((sum, item) => sum + Number(item.cost.original_cost_usd || 0), 0),
      optimized_cost_usd: priced.reduce((sum, item) => sum + Number(item.cost.optimized_cost_usd || 0), 0),
      estimated_savings_usd: priced.reduce((sum, item) => sum + Number(item.cost.estimated_savings_usd || 0), 0),
      model_pricing: 'Estimated Gemini standard paid-tier list prices; local Ollama requests excluded',
      requests_included: priced.length,
    } : null;
    const selectedEntry = entries.find((item) => item.id === selectedMessage?.id) || entries[entries.length - 1];

    return {
      summary: {
        total_queries: entries.length,
        token_reduction_percent: originalTokens ? (tokensSaved / originalTokens) * 100 : 0,
        similarity,
        tokens: { original_tokens: originalTokens, optimized_tokens: optimizedTokens, tokens_saved: tokensSaved },
        cost,
        strategy_used: selectedEntry.strategy_used,
        selected_llm: selectedEntry.selected_llm,
        target_achieved: selectedEntry.target_achieved,
        strategy_targets: {
          target_reduction: selectedMessage?.metrics?.target_reduction_percent || 30,
          min_similarity: selectedMessage?.metrics?.minimum_similarity || 0.85,
        },
        token_count_method: selectedEntry.metrics.token_count_method,
      },
      prompts: entries,
      raw_api_response: selectedMessage?.fullApiResponse || entries[entries.length - 1].fullApiResponse,
    };
  }

  static transformApiResponseToAnalytics(apiResponse) {
    if (!apiResponse?.data) return null;
    const syntheticMessage = {
      id: `prompt-${apiResponse.timestamp || Date.now()}`,
      type: 'assistant',
      fullApiResponse: apiResponse,
    };
    return this.transformMessagesToAnalytics([syntheticMessage], syntheticMessage);
  }

  static getMinSimilarityForStrategy(strategy) {
    return { aggressive: 0.80, balanced: 0.85, conservative: 0.90 }[strategy] || 0.85;
  }

  static getTargetReductionForStrategy(strategy) {
    return { aggressive: 35, balanced: 30, conservative: 15 }[strategy] || 30;
  }

  static prepareCostData(summary) {
    if (!summary?.cost) return [];
    return [
      { name: 'Original', cost: summary.cost.original_cost_usd },
      { name: 'Optimized', cost: summary.cost.optimized_cost_usd },
    ];
  }

  static calculateMetrics(summary) {
    if (!summary?.cost) return { totalSavings: '0.000000', savingsPercentage: '0.0' };
    const original = Number(summary.cost.original_cost_usd || 0);
    const optimized = Number(summary.cost.optimized_cost_usd || 0);
    return {
      totalSavings: (original - optimized).toFixed(6),
      savingsPercentage: original > 0 ? (((original - optimized) / original) * 100).toFixed(1) : '0.0',
    };
  }

  static async getStrategies() {
    try { return { success: true, data: await apiService.get('/api/strategies') }; }
    catch (error) { return { success: false, error: error.message }; }
  }

  static async checkHealth() {
    try { return { success: true, data: await apiService.get('/api/health') }; }
    catch (error) { return { success: false, error: error.message }; }
  }

  static async getStatus() {
    try { return { success: true, data: await apiService.get('/api/status') }; }
    catch (error) { return { success: false, error: error.message }; }
  }
}
