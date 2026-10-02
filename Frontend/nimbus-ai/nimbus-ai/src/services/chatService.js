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
  const lowerPrompt = prompt.trim().toLowerCase();
  const asksForCode = /\b(write|create|implement|build|debug|fix|refactor|review|generate|complete|optimize)\b/i.test(lowerPrompt);
  const namesCode = /\b(code|coding|programming|script|program|api endpoint|sql query)\b/i.test(lowerPrompt);
  const asksForCallable = /\b(write|create|implement|build|debug|fix|refactor|review|generate|complete)\b/i.test(lowerPrompt)
    && /\b(function|method|class)\b/i.test(lowerPrompt);

  if (
    lowerPrompt.includes('```') ||
    /\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|sql|bash|powershell)\b/i.test(lowerPrompt) ||
    (asksForCode && namesCode) ||
    asksForCallable
  ) {
    return 'coding';
  } else if (/\b(calculate|solve|compute|derive|prove|integrate|differentiate|equation|math|mathematics|integral|derivative)\b/i.test(lowerPrompt)) {
    return 'math';
  } else {
    return 'generic';
  }
}

// Determine prompt type for UI display
function getPromptType(prompt) {
  const lowerPrompt = prompt.toLowerCase();
  if (/\b(calculate|solve|compute|derive|prove|integrate|differentiate|equation|formula|math)\b/i.test(lowerPrompt)) {
    return PROMPT_TYPES.MATH;
  } else if (getPromptCategory(prompt) === 'coding') {
    return PROMPT_TYPES.CODE;
  } else if (lowerPrompt.includes('story') || lowerPrompt.includes('imagine')) {
    return PROMPT_TYPES.CREATIVE;
  } else {
    return PROMPT_TYPES.DEFAULT;
  }
}

export class ChatService {
  // Real API call to /api/process endpoint
  static async sendMessageToAPI({
    prompt,
    includeResponse = true,
    modelPreference = 'auto',
    selectedModel = null,
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

      // ✅ Fixed: was '/process', now '/api/process'
      const response = await apiService.post('/api/process', payload);

      console.log('API Response:', response);
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
