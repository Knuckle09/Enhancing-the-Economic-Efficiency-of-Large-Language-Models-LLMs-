// utils/helpers.js
import { PROMPT_TYPES } from '../data/constants.js';

// Generate unique IDs for messages
export const generateId = () => {
  return Date.now() + Math.random().toString(36).substr(2, 9);
};

// Determine prompt type based on content
export const getPromptType = (prompt) => {
  const lowerPrompt = prompt.toLowerCase();
  if (lowerPrompt.includes('story') || lowerPrompt.includes('creative') || lowerPrompt.includes('write')) {
    return PROMPT_TYPES.CREATIVE;
  } else if (lowerPrompt.includes('calculate') || lowerPrompt.includes('math') || lowerPrompt.includes('formula')) {
    return PROMPT_TYPES.MATH;
  } else if (lowerPrompt.includes('function') || lowerPrompt.includes('code') || lowerPrompt.includes('python')) {
    return PROMPT_TYPES.CODE;
  }
  return PROMPT_TYPES.DEFAULT;
};

// Format timestamps for display
export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleTimeString();
};

// Calculate cost savings percentage
export const calculateSavingsPercentage = (originalCost, optimizedCost) => {
  return ((1 - optimizedCost / originalCost) * 100).toFixed(1);
};

// Truncate text with ellipsis
export const truncateText = (text, maxLength) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
};