// Chat state, local conversation memory, and model feedback.

import React, { createContext, useContext, useEffect, useReducer } from 'react';
import { ChatService } from '../services/chatService.js';
import apiService from '../services/api.js';
import { MESSAGE_TYPES, OPTIMIZATION_STRATEGIES } from '../data/constants.js';

const CHAT_STORAGE_KEY = 'nimbus-ai-chat-v1';

function loadMessages() {
  try {
    const saved = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || '[]');
    return Array.isArray(saved)
      ? saved.map((message) => ({ ...message, timestamp: new Date(message.timestamp) }))
      : [];
  } catch {
    return [];
  }
}

const initialState = {
  messages: loadMessages(),
  isLoading: false,
  currentPrompt: '',
  error: null,
  selectedStrategy: OPTIMIZATION_STRATEGIES.BALANCED,
  modelPreference: 'auto',
  selectedModel: null,
};

const ACTIONS = {
  SET_PROMPT: 'SET_PROMPT',
  ADD_MESSAGE: 'ADD_MESSAGE',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_MESSAGES: 'CLEAR_MESSAGES',
  SET_STRATEGY: 'SET_STRATEGY',
  SET_MODEL_PREFERENCE: 'SET_MODEL_PREFERENCE',
  SET_SELECTED_MODEL: 'SET_SELECTED_MODEL',
  SET_FEEDBACK: 'SET_FEEDBACK',
};

function chatReducer(state, action) {
  switch (action.type) {
    case ACTIONS.SET_PROMPT:
      return { ...state, currentPrompt: action.payload };
    case ACTIONS.ADD_MESSAGE:
      return { ...state, messages: [...state.messages, action.payload] };
    case ACTIONS.SET_LOADING:
      return { ...state, isLoading: action.payload };
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
    case ACTIONS.CLEAR_MESSAGES:
      return { ...state, messages: [], error: null };
    case ACTIONS.SET_STRATEGY:
      return { ...state, selectedStrategy: action.payload };
    case ACTIONS.SET_MODEL_PREFERENCE:
      return { ...state, modelPreference: action.payload };
    case ACTIONS.SET_SELECTED_MODEL:
      return { ...state, selectedModel: action.payload };
    case ACTIONS.SET_FEEDBACK:
      return {
        ...state,
        messages: state.messages.map((message) => message.id === action.payload.id
          ? { ...message, feedback: action.payload.rating }
          : message),
      };
    default:
      return state;
  }
}

const ChatContext = createContext(null);

export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  useEffect(() => {
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(state.messages.slice(-60)));
    } catch (error) {
      console.warn('Could not save chat history in this browser:', error);
    }
  }, [state.messages]);

  const setCurrentPrompt = (prompt) => dispatch({ type: ACTIONS.SET_PROMPT, payload: prompt });
  const addMessage = (message) => dispatch({ type: ACTIONS.ADD_MESSAGE, payload: message });
  const setLoading = (loading) => dispatch({ type: ACTIONS.SET_LOADING, payload: loading });
  const setError = (error) => dispatch({ type: ACTIONS.SET_ERROR, payload: error });
  const clearMessages = () => dispatch({ type: ACTIONS.CLEAR_MESSAGES });
  const setStrategy = (strategy) => dispatch({ type: ACTIONS.SET_STRATEGY, payload: strategy });
  const setModelPreference = (value) => dispatch({ type: ACTIONS.SET_MODEL_PREFERENCE, payload: value });
  const setSelectedModel = (model) => dispatch({ type: ACTIONS.SET_SELECTED_MODEL, payload: model });

  const sendMessage = async (promptText) => {
    if (!promptText.trim() || state.isLoading) return;

    const history = state.messages
      .filter((message) => !message.isError && (message.type === MESSAGE_TYPES.USER || message.type === MESSAGE_TYPES.ASSISTANT))
      .slice(-12)
      .map((message) => ({
        role: message.type === MESSAGE_TYPES.USER ? 'user' : 'assistant',
        content: message.content,
      }));

    addMessage(ChatService.createMessage(MESSAGE_TYPES.USER, promptText));
    setCurrentPrompt('');
    setLoading(true);
    setError(null);

    try {
      const modelPreference = state.modelPreference || 'auto';
      const result = await ChatService.sendMessageToAPI({
        prompt: promptText,
        history,
        strategy: state.selectedStrategy,
        includeResponse: true,
        modelPreference,
        selectedModel: modelPreference === 'manual' ? state.selectedModel : null,
      });

      if (!result.success) throw new Error(result.error || 'Failed to process prompt');

      const apiResponse = result.data;
      const data = apiResponse.data || {};
      let assistantText;
      if (data.response_error) {
        assistantText = `Prompt optimization completed, but generation failed: ${data.response_error}`;
      } else if (data.response) {
        assistantText = `${data.model_fallback_notice ? `ℹ️ ${data.model_fallback_notice}\n\n` : ''}${data.cache_hit ? '⚡ Reused a highly similar recent answer.\n\n' : ''}${data.response}`;
      } else {
        assistantText = 'Prompt optimized successfully. No model response was requested.';
      }

      addMessage(ChatService.createMessage(MESSAGE_TYPES.ASSISTANT, assistantText, {
        model: data.selected_llm || 'Unknown',
        originalPrompt: data.original_prompt || promptText,
        optimizedPrompt: data.optimized_prompt || promptText,
        strategyUsed: data.strategy_used,
        tokenReductionPercent: Number(data.token_reduction_percent || 0),
        similarity: Number(data.similarity ?? 1),
        similarityMethod: data.similarity_method,
        category: data.category || 'generic',
        targetAchieved: data.target_achieved,
        metrics: data.metrics || {},
        processingTime: Number(apiResponse.processing_time || 0),
        timestamp: apiResponse.timestamp || new Date().toISOString(),
        cost: data.cost || null,
        cacheHit: Boolean(data.cache_hit),
        routingReason: data.routing_reason || '',
        actualUsage: data.actual_usage || null,
        fullApiResponse: apiResponse,
      }));
    } catch (error) {
      const errorMessage = error.message || 'An error occurred while sending the message';
      setError(errorMessage);
      addMessage(ChatService.createMessage(MESSAGE_TYPES.ASSISTANT, `Error: ${errorMessage}`, { isError: true }));
    } finally {
      setLoading(false);
    }
  };

  const rateResponse = async (message, rating) => {
    if (!message?.model || message.feedback) return;
    dispatch({ type: ACTIONS.SET_FEEDBACK, payload: { id: message.id, rating } });
    try {
      await apiService.post('/api/feedback', {
        model: message.model,
        category: message.category || 'generic',
        rating,
      });
    } catch (error) {
      console.warn('Could not send model feedback to the backend:', error.message);
    }
  };

  const value = {
    messages: state.messages,
    isLoading: state.isLoading,
    currentPrompt: state.currentPrompt,
    error: state.error,
    modelPreference: state.modelPreference,
    selectedModel: state.selectedModel,
    selectedStrategy: state.selectedStrategy,
    setCurrentPrompt,
    sendMessage,
    clearMessages,
    setModelPreference,
    setSelectedModel,
    setStrategy,
    rateResponse,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) throw new Error('useChat must be used within a ChatProvider');
  return context;
};
