// services/chatService.js
// Handles chat-related API calls

import { PROMPT_TYPES } from '../data/constants.js';
import apiService from './api.js';

// Generates a unique ID for messages
function generateId() {
  return '_' + Math.random().toString(36).substr(2, 9);
}

// Determine prompt category for API
function getPromptCategory(prompt) {
  const lowerPrompt = prompt.toLowerCase();

  if (
    lowerPrompt.includes('code') ||
    lowerPrompt.includes('function') ||
    lowerPrompt.includes('python') ||
    lowerPrompt.includes('javascript') ||
    lowerPrompt.includes('program')
  ) {
    return 'coding';
  } else if (
    lowerPrompt.includes('calculate') ||
    lowerPrompt.includes('formula') ||
    lowerPrompt.includes('math') ||
    lowerPrompt.includes('equation')
  ) {
    return 'math';
  } else {
    return 'generic';
  }
}

// Determine prompt type for UI display
function getPromptType(prompt) {
  const lowerPrompt = prompt.toLowerCase();

  if (lowerPrompt.includes('calculate') || lowerPrompt.includes('formula')) {
    return PROMPT_TYPES.MATH;
  } else if (lowerPrompt.includes('code') || lowerPrompt.includes('function')) {
    return PROMPT_TYPES.CODE;
  } else if (lowerPrompt.includes('story') || lowerPrompt.includes('imagine')) {
    return PROMPT_TYPES.CREATIVE;
  } else {
    return PROMPT_TYPES.DEFAULT;
  }
}

export class ChatService {
  // Real API call to /process endpoint
  static async sendMessageToAPI({
    prompt,
    includeResponse = true,
    modelPreference = 'auto',   // 'auto' or 'manual'
    selectedModel = null,       // only used when manual
  }) {
    try {
      const payload = {
        prompt: prompt.trim(),
        include_response: includeResponse,
        model_preference: modelPreference,
      };

      if (modelPreference === 'manual' && selectedModel) {
        payload.selected_model = selectedModel;
      }

      console.log('Sending to API:', payload);
      const response = await apiService.post('/process', payload);
      console.log('API Response:', response);

      // response is exactly the backend contract
      return { success: true, data: response };
    } catch (error) {
      console.error('API Error:', error);
      return { success: false, error: error.message };
    }
  }

  // Utility to create a chat message
  static createMessage(type, content, additionalData = {}) {
    return {
      id: generateId(),
      type,
      content,
      timestamp: new Date(),
      ...additionalData,
    };
  }

  // Get prompt type for UI categorization
  static getPromptType(prompt) {
    return getPromptType(prompt);
  }

  // Get category for API
  static getPromptCategory(prompt) {
    return getPromptCategory(prompt);
  }
}