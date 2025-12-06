// components/analytics/SummaryCards.jsx
import React from 'react';
import { Target, Activity, Zap, TrendingDown } from 'lucide-react';

const SummaryCards = ({ summary }) => {
  if (!summary) return null;

  // Safely convert values to numbers
  const tokenReduction = Number(
  summary.token_reduction_percent ?? 0
);

const similarityScore = Number(
  summary.similarity ?? 0           // already in %
);

const tokensSaved = Number(
  summary.tokens?.tokens_saved ?? 0
);


  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {/* Token Reduction Card */}
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 border-l-4 border-blue-500">
        <div className="flex items-center justify-between mb-3">
          <div className="bg-blue-600 p-3 rounded-lg">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div className={`text-xs px-2 py-1 rounded ${
            tokenReduction >= 20 ? 'bg-green-900/50 text-green-300' : 'bg-yellow-900/50 text-yellow-300'
          }`}>
            {tokenReduction >= 20 ? 'Excellent' : 'Good'}
          </div>
        </div>
        <h3 className="text-sm font-medium text-gray-400 mb-1">Token Reduction</h3>
        <p className="text-3xl font-bold text-blue-400">
          {isNaN(tokenReduction) ? 'N/A' : tokenReduction.toFixed(1)}%
        </p>
        <p className="text-gray-500 text-xs mt-2">Efficiency improvement</p>
      </div>

      {/* Similarity Score Card */}
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 border-l-4 border-green-500">
        <div className="flex items-center justify-between mb-3">
          <div className="bg-green-600 p-3 rounded-lg">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div className={`text-xs px-2 py-1 rounded ${
            similarityScore >= 90 ? 'bg-green-900/50 text-green-300' : 'bg-yellow-900/50 text-yellow-300'
          }`}>
            {similarityScore >= 90 ? 'High Quality' : 'Acceptable'}
          </div>
        </div>
        <h3 className="text-sm font-medium text-gray-400 mb-1">Similarity Score</h3>
        <p className="text-3xl font-bold text-green-400">
          {isNaN(similarityScore) ? 'N/A' : similarityScore.toFixed(1)}%
        </p>
        <p className="text-gray-500 text-xs mt-2">Quality maintained</p>
      </div>

      {/* Tokens Saved Card */}
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 border-l-4 border-purple-500">
        <div className="flex items-center justify-between mb-3">
          <div className="bg-purple-600 p-3 rounded-lg">
            <TrendingDown className="w-6 h-6 text-white" />
          </div>
          <div className="text-xs px-2 py-1 rounded bg-purple-900/50 text-purple-300">
            Saved
          </div>
        </div>
        <h3 className="text-sm font-medium text-gray-400 mb-1">Tokens Saved</h3>
        <p className="text-3xl font-bold text-purple-400">
          {isNaN(tokensSaved) ? 'N/A' : tokensSaved}
        </p>
        <p className="text-gray-500 text-xs mt-2">Total reduction</p>
      </div>
    </div>
  );
};

export default SummaryCards;
