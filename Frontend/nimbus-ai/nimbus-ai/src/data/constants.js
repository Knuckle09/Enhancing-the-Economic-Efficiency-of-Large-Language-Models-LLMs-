// data/constants.js
// Application constants

export const PROMPT_TYPES = {
  CREATIVE: 'creative',
  MATH: 'math',
  CODE: 'code',
  DEFAULT: 'default'
};

export const MESSAGE_TYPES = {
  USER: 'user',
  ASSISTANT: 'assistant'
};

export const VIEW_TYPES = {
  CHAT: 'chat',
  ANALYTICS: 'analytics'
};

export const MODELS = {
  CLAUDE_SONNET: 'Claude-3.5-Sonnet',
  GPT4_TURBO: 'GPT-4-Turbo'
};

// API-related constants
export const OPTIMIZATION_STRATEGIES = {
  AGGRESSIVE: 'aggressive',
  BALANCED: 'balanced',
  CONSERVATIVE: 'conservative'
};

export const PROMPT_CATEGORIES = {
  CODING: 'coding',
  MATH: 'math',
  GENERIC: 'generic'
};

// Strategy descriptions (from your API /strategies endpoint)
export const STRATEGY_DESCRIPTIONS = {
  aggressive: {
    label: 'Aggressive',
    description: '35% token reduction, 85% similarity',
    target_reduction: 35,
    min_similarity: 0.85,
  },
  balanced: {
    label: 'Balanced',
    description: '30% token reduction, 85% similarity',
    target_reduction: 30,
    min_similarity: 0.85,
  },
  conservative: {
    label: 'Conservative',
    description: '15% token reduction, 90% similarity',
    target_reduction: 15,
    min_similarity: 0.90,
  },
};

// LLM mappings (from your API /strategies endpoint)
export const LLM_MAPPINGS = {
  coding: 'codellama:7b',
  math: 'qwen2-math:7b',
  generic: 'tinyllama:latest',
};

export const UI_CONSTANTS = {
  MAX_MESSAGE_WIDTH: 'max-w-4xl'
};