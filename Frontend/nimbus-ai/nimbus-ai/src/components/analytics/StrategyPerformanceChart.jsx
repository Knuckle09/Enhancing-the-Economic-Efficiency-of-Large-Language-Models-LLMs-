import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const StrategyPerformanceChart = ({ prompts = [] }) => {
  const groups = prompts.reduce((result, prompt) => {
    const name = prompt.strategy_used || 'unknown';
    result[name] ||= { count: 0, reduction: 0, similarity: 0, similarityCount: 0 };
    result[name].count += 1;
    result[name].reduction += Number(prompt.metrics?.token_reduction_percent || 0);
    if (Number.isFinite(prompt.metrics?.similarity)) {
      result[name].similarity += prompt.metrics.similarity * 100;
      result[name].similarityCount += 1;
    }
    return result;
  }, {});
  const data = Object.entries(groups).map(([strategy, value]) => ({
    strategy: strategy[0].toUpperCase() + strategy.slice(1),
    'Token reduction': Number((value.reduction / value.count).toFixed(1)),
    'Semantic similarity': value.similarityCount ? Number((value.similarity / value.similarityCount).toFixed(1)) : null,
    replies: value.count,
  }));

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold text-gray-200 mb-2">Strategy Performance</h3>
      <p className="text-xs text-gray-400 mb-3">Live averages; semantic scores appear when embeddings are available</p>
      {data.length ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="strategy" stroke="#9ca3af" />
              <YAxis domain={[0, 100]} stroke="#9ca3af" />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
              <Legend />
              <Bar dataKey="Token reduction" fill="#3b82f6" radius={[5, 5, 0, 0]} />
              <Bar dataKey="Semantic similarity" fill="#10b981" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : <div className="h-64 flex items-center justify-center text-gray-500">No strategy data yet</div>}
    </div>
  );
};

export default StrategyPerformanceChart;
