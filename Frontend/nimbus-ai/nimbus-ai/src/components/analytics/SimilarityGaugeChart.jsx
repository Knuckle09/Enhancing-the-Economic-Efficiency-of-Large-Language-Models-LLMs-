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

  const measured = Number.isFinite(data.similarity);
  const similarity = measured ? data.similarity : null;
  const minSimilarity = data.strategy_targets?.min_similarity ?? 0.90;
  const similarityPercent = measured ? (similarity * 100).toFixed(1) : 'N/A';
  const targetPercent = (minSimilarity * 100).toFixed(0);
  const targetAchieved = measured && similarity >= minSimilarity;
  
  const gaugeData = [
    {
      name: 'Similarity',
      value: measured ? Number(similarityPercent) : 0,
      fill: targetAchieved ? '#10b981' : '#f59e0b'
    }
  ];

  return (
    <div className="bg-gray-800 rounded-xl shadow-2xl p-6 border-t-4 border-green-600">
      <div className="flex items-center space-x-2 mb-6">
        <Target className="w-6 h-6 text-green-400" />
        <h3 className="text-xl font-semibold text-gray-200">Semantic Similarity</h3>
      </div>
      <p className="text-xs text-gray-400 -mt-4 mb-3">Original prompt vs. optimized prompt · {data.similarity_method || 'embedding score unavailable'}</p>
      
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
              {measured ? `${similarityPercent}%` : 'N/A'}
            </text>
            <text
              x="50%"
              y="60%"
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-sm"
              fill="#9ca3af"
            >
              Embedding cosine
            </text>
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 flex items-center justify-between bg-gray-700/50 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          {targetAchieved ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : measured ? (
            <Activity className="w-5 h-5 text-yellow-400" />
          ) : <Activity className="w-5 h-5 text-gray-500" />}
          <span className="text-gray-300 text-sm">Target: {targetPercent}%</span>
        </div>
        <div className={`font-semibold text-sm ${
          targetAchieved
            ? 'text-green-400' 
            : measured ? 'text-yellow-400' : 'text-gray-400'
        }`}>
          {targetAchieved ? '✓ Target Achieved' : measured ? '⚠ Below Target' : 'Not measured'}
        </div>
      </div>
    </div>
  );
};

export default SimilarityGaugeChart;
