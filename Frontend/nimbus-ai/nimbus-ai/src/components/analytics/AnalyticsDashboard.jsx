// components/analytics/AnalyticsDashboard.jsx
import React from 'react';
import { useAnalytics } from '../../hooks/useAnalytics.js';
import { useAppState } from '../../contexts/AppStateContext.jsx';
import { truncateText } from '../../utils/helpers.js';
import SummaryCards from './SummaryCards.jsx';
import CostChart from './CostChart.jsx';
import TokenReductionChart from './TokenReductionChart.jsx';
import SimilarityGaugeChart from './SimilarityGaugeChart.jsx';
import PromptDetailsCard from './PromptDetailsCard.jsx';

const AnalyticsDashboard = () => {
  const { 
    analyticsData, 
    currentMessage,
    expandedCards, 
    toggleCardExpansion,
    isAnalyzing, 
    error 
  } = useAnalytics();

  const { switchToChat } = useAppState();

  const handleBackToChat = () => {
    switchToChat();
  };

  if (isAnalyzing) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <h2 className="text-2xl font-bold text-gray-400">Processing analytics...</h2>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 mb-4">
            <h2 className="text-2xl font-bold text-red-500 mb-2">Error</h2>
            <p className="text-red-300">{error}</p>
          </div>
          <button
            onClick={handleBackToChat}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white px-6 py-3 rounded-lg transition-colors"
          >
            ← Back to Chat
          </button>
        </div>
      </div>
    );
  }

  if (!analyticsData || !analyticsData.summary) {
    return (
      <div className="min-h-screen bg-gray-900 text-gray-100 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-400 mb-4">No Analytics Data Available</h2>
          <p className="text-gray-500 mb-6">Please analyze a message from the chat to see analytics.</p>
          <button
            onClick={handleBackToChat}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white px-6 py-3 rounded-lg transition-colors"
          >
            ← Back to Chat
          </button>
        </div>
      </div>
    );
  }

  const { summary, prompts } = analyticsData;

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Analytics Dashboard
            </h1>
            {currentMessage && currentMessage.originalPrompt && (
              <p className="text-gray-400">
                Analysis for: "{truncateText(currentMessage.originalPrompt, 60)}"
              </p>
            )}
          </div>
          <button
            onClick={handleBackToChat}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
          >
            <span>←</span>
            <span>Back to Chat</span>
          </button>
        </div>

        {/* Summary Cards */}
        <SummaryCards summary={summary} />

        {/* Charts Grid - 2 Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <TokenReductionChart data={summary}  />
          <SimilarityGaugeChart data={summary} />
          <CostChart summary={summary} />
        </div>

        {/* Prompt Detail Cards */}
        {prompts && prompts.length > 0 && (
          <div className="space-y-6 mt-8">
            <h3 className="text-2xl font-bold text-gray-200 mb-4">Prompt Analysis Details</h3>
            {prompts.map((prompt) => (
              <PromptDetailsCard
                key={prompt.id}
                prompt={prompt}
                isExpanded={expandedCards[prompt.id]}
                onToggle={toggleCardExpansion}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsDashboard;