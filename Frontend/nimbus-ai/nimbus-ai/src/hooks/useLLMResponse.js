// hooks/useLLMResponse.js
// Custom hook for handling LLM response logic with strategy support

import { useState, useCallback } from 'react';
import { ChatService } from '../services/chatService.js';

export const useLLMResponse = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendLLMRequest = useCallback(async (prompt, options = {}) => {
    if (!prompt?.trim()) return null;

    setIsLoading(true);
    setError(null);

    try {
      // Pass strategy and category options to the API
      const response = await ChatService.sendMessageToAPI(prompt, {
        strategy: options.strategy || 'balanced',
        category: options.category || null, // auto-detect if null
        use_mock_llm: options.use_mock_llm || false,
      });

      if (!response.success) {
        throw new Error(response.error || 'Failed to get response');
      }

      return response;
    } catch (err) {
      const errorMessage = err.message || 'An error occurred';
      setError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    isLoading,
    error,
    sendLLMRequest,
  };
};