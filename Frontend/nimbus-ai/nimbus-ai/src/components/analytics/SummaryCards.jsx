import React from 'react';
import { Activity, Clock, DollarSign, MessageSquare, TrendingDown, Zap } from 'lucide-react';

const SummaryCards = ({ summary }) => {
  if (!summary) return null;

  const reduction = Number(summary.token_reduction_percent || 0);
  const similarity = Number.isFinite(summary.similarity) ? `${(summary.similarity * 100).toFixed(1)}%` : 'N/A';
  const savings = Number.isFinite(summary.cost?.estimated_savings_percent)
    ? `${summary.cost.estimated_savings_percent.toFixed(1)}%`
    : 'N/A';
  const tokensSaved = Number(summary.tokens?.tokens_saved || 0);
  const cards = [
    {
      label: 'Token Reduction',
      value: `${reduction.toFixed(1)}%`,
      note: 'Live session average',
      icon: Zap,
      color: 'blue',
    },
    {
      label: 'Semantic Similarity',
      value: similarity,
      note: `${summary.semantic_similarity_samples || 0} embedding scores`,
      icon: Activity,
      color: 'green',
    },
    {
      label: 'Estimated Cost Savings',
      value: savings,
      note: summary.cost ? `${summary.cost.requests_included} Gemini requests` : 'Gemini cost unavailable',
      icon: DollarSign,
      color: 'yellow',
    },
    {
      label: 'Tokens Saved',
      value: tokensSaved.toLocaleString(),
      note: 'Live session total',
      icon: TrendingDown,
      color: 'purple',
    },
    {
      label: 'Avg. Processing Time',
      value: `${Number(summary.average_processing_time || 0).toFixed(2)}s`,
      note: 'End-to-end API time',
      icon: Clock,
      color: 'blue',
    },
    {
      label: 'Analyzed Replies',
      value: String(summary.total_queries || 0),
      note: `${summary.multi_turn_queries || 0} with conversation history`,
      icon: MessageSquare,
      color: 'cyan',
    },
  ];

  const colors = {
    blue: ['border-blue-500', 'bg-blue-600', 'text-blue-400'],
    green: ['border-green-500', 'bg-green-600', 'text-green-400'],
    yellow: ['border-yellow-500', 'bg-yellow-600', 'text-yellow-400'],
    purple: ['border-purple-500', 'bg-purple-600', 'text-purple-400'],
    cyan: ['border-cyan-500', 'bg-cyan-700', 'text-cyan-400'],
  };

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold text-gray-200 mb-4">Live Session Metrics</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6 gap-4">
        {cards.map(({ label, value, note, icon, color }) => (
          <div key={label} className={`bg-gray-800 rounded-xl shadow-lg p-5 border-l-4 ${colors[color][0]}`}>
            <div className="flex items-center justify-between mb-3">
              <div className={`${colors[color][1]} p-2 rounded-lg`}>{React.createElement(icon, { className: 'w-5 h-5 text-white' })}</div>
              {label === 'Estimated Cost Savings' && summary.cost && (
                <span className={`text-xs px-2 py-1 rounded ${summary.cost.estimated_savings_percent >= 0 ? 'bg-green-900/50 text-green-300' : 'bg-red-900/50 text-red-300'}`}>
                  {summary.cost.estimated_savings_percent >= 0 ? 'Savings' : 'Cost increase'}
                </span>
              )}
            </div>
            <h3 className="text-sm font-medium text-gray-400 mb-1">{label}</h3>
            <p className={`text-2xl font-bold ${colors[color][2]}`}>{value}</p>
            <p className="text-gray-500 text-xs mt-2">{note}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SummaryCards;
