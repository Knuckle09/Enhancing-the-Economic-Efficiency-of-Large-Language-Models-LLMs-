// contexts/AnalyticsContext.jsx
// Manages analytics state and functionality - uses data from chat messages

import React, { createContext, useContext, useReducer } from 'react';
import { AnalyticsService } from '../services/analyticsService.js';
import { useAppState } from './AppStateContext.jsx';

// Initial state
const initialState = {
  analyticsData: null,
  isAnalyzing: false,
  currentMessage: null, // stores the full message object
  expandedCards: {},
  error: null,
};

// Action types
const ACTIONS = {
  SET_ANALYZING: 'SET_ANALYZING',
  SET_ANALYTICS_DATA: 'SET_ANALYTICS_DATA',
  SET_CURRENT_MESSAGE: 'SET_CURRENT_MESSAGE',
  TOGGLE_CARD: 'TOGGLE_CARD',
  RESET_EXPANDED_CARDS: 'RESET_EXPANDED_CARDS',
  SET_ERROR: 'SET_ERROR',
};

// Reducer
const analyticsReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_ANALYZING:
      return { ...state, isAnalyzing: action.payload };
    
    case ACTIONS.SET_ANALYTICS_DATA:
      return { ...state, analyticsData: action.payload };
    
    case ACTIONS.SET_CURRENT_MESSAGE:
      return { ...state, currentMessage: action.payload };
    
    case ACTIONS.TOGGLE_CARD:
      return {
        ...state,
        expandedCards: {
          ...state.expandedCards,
          [action.payload]: !state.expandedCards[action.payload],
        },
      };
    
    case ACTIONS.RESET_EXPANDED_CARDS:
      return { ...state, expandedCards: {} };
    
    case ACTIONS.SET_ERROR:
      return { ...state, error: action.payload };
    
    default:
      return state;
  }
};

// Create context
const AnalyticsContext = createContext();

// Provider component
export const AnalyticsProvider = ({ children }) => {
  const [state, dispatch] = useReducer(analyticsReducer, initialState);
  const { switchToAnalytics } = useAppState();

  // Actions
  const setAnalyzing = (isAnalyzing) => dispatch({ type: ACTIONS.SET_ANALYZING, payload: isAnalyzing });
  const setAnalyticsData = (data) => dispatch({ type: ACTIONS.SET_ANALYTICS_DATA, payload: data });
  const setCurrentMessage = (message) => dispatch({ type: ACTIONS.SET_CURRENT_MESSAGE, payload: message });
  const toggleCardExpansion = (promptId) => dispatch({ type: ACTIONS.TOGGLE_CARD, payload: promptId });
  const resetExpandedCards = () => dispatch({ type: ACTIONS.RESET_EXPANDED_CARDS });
  const setError = (error) => dispatch({ type: ACTIONS.SET_ERROR, payload: error });

  // Analyze message - uses stored API response data
  const analyzeMessage = async (message) => {
    if (!message || !message.fullApiResponse) {
      setError('No analytics data available for this message');
      return;
    }

    try {
      setCurrentMessage(message);
      setAnalyzing(true);
      setError(null);

      // Transform the stored API response to analytics format
      const analyticsData = AnalyticsService.transformApiResponseToAnalytics(
        message.fullApiResponse
      );

      if (analyticsData) {
        setAnalyticsData(analyticsData);
        switchToAnalytics();
      } else {
        setError('Failed to process analytics data');
      }
    } catch (error) {
      setError('An error occurred while processing analytics');
      console.error('Analytics error:', error);
    } finally {
      setAnalyzing(false);
    }
  };

  const value = {
    // State
    analyticsData: state.analyticsData,
    isAnalyzing: state.isAnalyzing,
    currentMessage: state.currentMessage,
    expandedCards: state.expandedCards,
    error: state.error,
    
    // Actions
    analyzeMessage,
    toggleCardExpansion,
    resetExpandedCards,
  };

  return (
    <AnalyticsContext.Provider value={value}>
      {children}
    </AnalyticsContext.Provider>
  );
};

// Custom hook
export const useAnalytics = () => {
  const context = useContext(AnalyticsContext);
  if (!context) throw new Error('useAnalytics must be used within an AnalyticsProvider');
  return context;
};