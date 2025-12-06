// contexts/ChatContext.jsx
// Manages chat state and provides chat-related functionality with real API integration

import React, { createContext, useContext, useReducer } from 'react';
import { ChatService } from '../services/chatService.js';
import { MESSAGE_TYPES, OPTIMIZATION_STRATEGIES } from '../data/constants.js';

// Initial state
const initialState = {
  messages: [],
  isLoading: false,
  currentPrompt: '',
  error: null,
  selectedStrategy: OPTIMIZATION_STRATEGIES.BALANCED, // default strategy
  modelPreference: 'auto',   // ✅ NEW
  selectedModel: null,       // ✅ NEW (used only if manual)
};

// Action types
const ACTIONS = {
  SET_PROMPT: 'SET_PROMPT',
  ADD_MESSAGE: 'ADD_MESSAGE',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_MESSAGES: 'CLEAR_MESSAGES',
  SET_STRATEGY: 'SET_STRATEGY',
  SET_MODEL_PREFERENCE: 'SET_MODEL_PREFERENCE',
  SET_SELECTED_MODEL: 'SET_SELECTED_MODEL',
};

// Reducer
const chatReducer = (state, action) => {
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
      return { ...state, messages: [] };
    
    case ACTIONS.SET_STRATEGY:
      return { ...state, selectedStrategy: action.payload };

    case ACTIONS.SET_MODEL_PREFERENCE:
      return { ...state, modelPreference: action.payload };

    case ACTIONS.SET_SELECTED_MODEL:
      return { ...state, selectedModel: action.payload };

    default:
      return state;
  }
};

// Create context
const ChatContext = createContext();

// Provider component
export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Actions
  const setCurrentPrompt = (prompt) => dispatch({ type: ACTIONS.SET_PROMPT, payload: prompt });
  const addMessage = (message) => dispatch({ type: ACTIONS.ADD_MESSAGE, payload: message });
  const setLoading = (loading) => dispatch({ type: ACTIONS.SET_LOADING, payload: loading });
  const setError = (error) => dispatch({ type: ACTIONS.SET_ERROR, payload: error });
  const clearMessages = () => dispatch({ type: ACTIONS.CLEAR_MESSAGES });
  const setStrategy = (strategy) => dispatch({ type: ACTIONS.SET_STRATEGY, payload: strategy });
  const setModelPreference = (value) => dispatch({ type: ACTIONS.SET_MODEL_PREFERENCE, payload: value });

  const setSelectedModel = (model) => dispatch({ type: ACTIONS.SET_SELECTED_MODEL, payload: model });


  // Send message using real API
  const sendMessage = async (promptText, options = {}) => {
  if (!promptText.trim()) return;

  // show user message
  const userMessage = ChatService.createMessage(MESSAGE_TYPES.USER, promptText);
  addMessage(userMessage);

  setCurrentPrompt('');
  setLoading(true);
  setError(null);

  try {
    // modelPreference & selectedModel can later come from UI
    const modelPreference = state.modelPreference || 'auto';
const selectedModel =
  modelPreference === 'manual' ? state.selectedModel : null;

const includeResponse = true;


    const result = await ChatService.sendMessageToAPI({
      prompt: promptText,
      includeResponse,
      modelPreference,
      selectedModel,
    });

    if (!result.success) {
      throw new Error(result.error || 'Failed to process prompt');
    }

    const apiResponse = result.data;   // full backend response
    const data = apiResponse.data;     // inner "data" object

    let assistantText = '';
    let isWarning = false;

    if (data.response_error) {
      // Optimization succeeded, LLM call failed
      assistantText = `⚠️ LLM error, but prompt was optimized.\n${data.response_error}`;
      isWarning = true;
    } else if (data.response) {
      assistantText = data.response;
    } else {
      assistantText = '✅ Prompt optimized successfully (no LLM response).';
    }

    const assistantMessage = ChatService.createMessage(
      MESSAGE_TYPES.ASSISTANT,
      assistantText,
      {
        isWarning,
        model: data.selected_llm || 'Unknown',
        originalPrompt: data.original_prompt || promptText,
        optimizedPrompt: data.optimized_prompt || promptText,
        strategyUsed: data.strategy_used,
        tokenReductionPercent: data.token_reduction_percent,
        similarity: data.similarity,
        category: data.category || 'generic',
        targetAchieved: data.target_achieved,
        metrics: data.metrics || {},
        processingTime: apiResponse.processing_time || 0,
        timestamp: apiResponse.timestamp || new Date().toISOString(),
        cost: data.cost || {},
        // full response for analytics
        fullApiResponse: apiResponse,
      }
    );

    addMessage(assistantMessage);
  } catch (err) {
    console.error('Send message error:', err);
    const errorMsg = err.message || 'An error occurred while sending the message';
    setError(errorMsg);

    const errorMessage = ChatService.createMessage(
      MESSAGE_TYPES.ASSISTANT,
      `Error: ${errorMsg}`,
      { isError: true }
    );
    addMessage(errorMessage);
  } finally {
    setLoading(false);
  }
};


  const value = {
  messages: state.messages,
  isLoading: state.isLoading,
  currentPrompt: state.currentPrompt,
  error: state.error,

  modelPreference: state.modelPreference,
  selectedModel: state.selectedModel,

  setCurrentPrompt,
  sendMessage,
  clearMessages,
  setModelPreference,
  setSelectedModel,
};


  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) throw new Error('useChat must be used within a ChatProvider');
  return context;
};