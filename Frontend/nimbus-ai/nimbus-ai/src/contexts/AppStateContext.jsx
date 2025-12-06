// contexts/AppStateContext.jsx
// Manages global app state (current view, navigation, etc.)

import React, { createContext, useContext, useState } from 'react';
import { VIEW_TYPES } from '../data/constants.js';

const AppStateContext = createContext();

export const AppStateProvider = ({ children }) => {
  const [currentView, setCurrentView] = useState(VIEW_TYPES.CHAT);

  const switchToChat = () => setCurrentView(VIEW_TYPES.CHAT);
  const switchToAnalytics = () => setCurrentView(VIEW_TYPES.ANALYTICS);

  const value = {
    currentView,
    switchToChat,
    switchToAnalytics
  };

  return (
    <AppStateContext.Provider value={value}>
      {children}
    </AppStateContext.Provider>
  );
};

export const useAppState = () => {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error('useAppState must be used within an AppStateProvider');
  }
  return context;
};