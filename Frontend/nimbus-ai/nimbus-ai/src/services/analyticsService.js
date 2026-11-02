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
    history_turns: Number(data.history_turns || 0),
    cache_hit: Boolean(data.cache_hit),
    feedback: message.feedback || null,
    metrics: {
      token_reduction_percent: Number(data.token_reduction_percent || 0),
      similarity: data.similarity == null ? null : Number(data.similarity),
      similarity_method: data.similarity_method || data.metrics?.similarity_method || 'not measured',
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
    const selectedEntry = entries.find((item) => item.id === selectedMessage?.id) || entries[entries.length - 1];
    const availableSimilarityMethods = entries
      .filter((item) => Number.isFinite(item.metrics.similarity) && item.metrics.similarity_method !== 'exact text match')
      .map((item) => item.metrics.similarity_method);
    const selectedSimilarityMethod = selectedEntry.metrics.similarity_method === 'exact text match'
      ? (availableSimilarityMethods[availableSimilarityMethods.length - 1] || 'exact text match')
      : selectedEntry.metrics.similarity_method;
    const semanticScores = entries.filter((item) => Number.isFinite(item.metrics.similarity)
      && (item.metrics.similarity_method === selectedSimilarityMethod || item.metrics.similarity_method === 'exact text match'));
    const similarity = semanticScores.length
      ? semanticScores.reduce((sum, item) => sum + item.metrics.similarity, 0) / semanticScores.length
      : null;
    const priced = entries.filter((item) => item.cost && Number.isFinite(Number(item.cost.optimized_cost_usd)));
    const cost = priced.length ? (() => {
      const originalCost = priced.reduce((sum, item) => sum + Number(item.cost.original_cost_usd || 0), 0);
      const optimizedCost = priced.reduce((sum, item) => sum + Number(item.cost.optimized_cost_usd || 0), 0);
      const savings = originalCost - optimizedCost;
      return {
        original_cost_usd: originalCost,
        optimized_cost_usd: optimizedCost,
        estimated_savings_usd: savings,
        estimated_savings_percent: originalCost > 0 ? (savings / originalCost) * 100 : 0,
        model_pricing: 'Estimated Gemini standard paid-tier list prices; local Ollama requests excluded',
        cost_basis: 'Same selected Gemini model; raw versus optimized prompt context, with semantic embedding and cache costs included',
        requests_included: priced.length,
      };
    })() : null;
    const averageProcessingTime = entries.reduce((sum, item) => sum + item.processing_time, 0) / entries.length;

    return {
      summary: {
        total_queries: entries.length,
        token_reduction_percent: originalTokens ? (tokensSaved / originalTokens) * 100 : 0,
        similarity,
        similarity_method: selectedSimilarityMethod,
        semantic_similarity_samples: semanticScores.length,
        tokens: { original_tokens: originalTokens, optimized_tokens: optimizedTokens, tokens_saved: tokensSaved },
        cost,
        average_processing_time: averageProcessingTime,
        cache_hits: entries.filter((item) => item.cache_hit).length,
        multi_turn_queries: entries.filter((item) => item.history_turns > 0).length,
        strategy_used: selectedEntry.strategy_used,
        selected_llm: selectedEntry.selected_llm,
        target_achieved: selectedEntry.target_achieved,
        strategy_targets: {
          target_reduction: selectedMessage?.metrics?.target_reduction_percent || 30,
          min_similarity: selectedMessage?.metrics?.minimum_similarity || 0.90,
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
    return { aggressive: 0.90, balanced: 0.90, conservative: 0.90 }[strategy] || 0.90;
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
      totalSavings: (original - optimized).toFixed(8),
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
