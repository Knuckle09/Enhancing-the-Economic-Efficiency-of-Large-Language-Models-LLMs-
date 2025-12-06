// Pages/Home.jsx
// Main page component - now clean and focused

import React from 'react';
import { ChatProvider } from '../contexts/ChatContext.jsx';
import { AnalyticsProvider } from '../contexts/AnalyticsContext.jsx';
import { AppStateProvider, useAppState } from '../contexts/AppStateContext.jsx';
import ChatInterface from '../components/chat/ChatInterface.jsx';
import AnalyticsDashboard from '../components/analytics/AnalyticsDashboard.jsx';
import { VIEW_TYPES } from '../data/constants.js';

// Inner component that uses the context
const HomeContent = () => {
  const { currentView } = useAppState();
  
  return currentView === VIEW_TYPES.ANALYTICS ? <AnalyticsDashboard /> : <ChatInterface />;
};

const Home = () => {
  return (
    <AppStateProvider>
      <ChatProvider>
        <AnalyticsProvider>
          <HomeContent />
        </AnalyticsProvider>
      </ChatProvider>
    </AppStateProvider>
  );
};

export default Home;