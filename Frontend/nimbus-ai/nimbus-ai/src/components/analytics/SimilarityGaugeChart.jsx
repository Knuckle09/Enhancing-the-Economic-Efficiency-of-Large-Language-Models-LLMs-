// components/analytics/SimilarityGaugeChart.jsx
import React from 'react';
import { Target, CheckCircle, Activity } from 'lucide-react';
import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts';

const SimilarityGaugeChart = ({ data }) => {
  if (!data) {
    return (
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6">
        <div className="flex items-center space-x-2 mb-6">
          <Target className="w-6 h-6 text-green-400" />
          <h3 className="text-xl font-semibold text-gray-200">Semantic Similarity</h3>
        </div>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No similarity data available
        </div>
      </div>
    );
  }

  const similarity = data.similarity || 0;
  const minSimilarity = data.strategy_targets?.min_similarity || 0.85;
  
  const similarityPercent = (similarity * 100).toFixed(1);
  const targetPercent = (minSimilarity * 100).toFixed(0);
  
  const gaugeData = [
    {
      name: 'Similarity',
      value: parseFloat(similarityPercent),
      fill: similarity >= minSimilarity ? '#10b981' : '#f59e0b'
    }
  ];

  return (
    <div className="bg-gray-800 rounded-xl shadow-2xl p-6 border-t-4 border-green-600">
      <div className="flex items-center space-x-2 mb-6">
        <Target className="w-6 h-6 text-green-400" />
        <h3 className="text-xl font-semibold text-gray-200">Semantic Similarity</h3>
      </div>
      
      <div className="h-64 flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart 
            cx="50%" 
            cy="50%" 
            innerRadius="60%" 
            outerRadius="90%" 
            data={gaugeData}
            startAngle={180}
            endAngle={0}
          >
            <RadialBar
              background
              dataKey="value"
              cornerRadius={10}
              fill={gaugeData[0].fill}
            />
            <text
              x="50%"
              y="45%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-4xl font-bold"
              fill="#f3f4f6"
            >
              {similarityPercent}%
            </text>
            <text
              x="50%"
              y="60%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-sm"
              fill="#9ca3af"
            >
              Similarity Score
            </text>
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          {similarity >= minSimilarity ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <Activity className="w-5 h-5 text-yellow-400" />
          )}
          <span className="text-gray-300 text-sm">Target: {targetPercent}%</span>
        </div>
        <div className={`font-semibold text-sm ${
          similarity >= minSimilarity 
            ? 'text-green-400' 
            : 'text-yellow-400'
        }`}>
          {similarity >= minSimilarity 
            ? '✓ Target Achieved' 
            : '⚠ Below Target'}
        </div>
      </div>
    </div>
  );
};

export default SimilarityGaugeChart;