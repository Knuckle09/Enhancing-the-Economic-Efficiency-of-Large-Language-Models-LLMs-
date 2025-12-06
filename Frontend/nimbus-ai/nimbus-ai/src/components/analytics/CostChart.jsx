// components/analytics/CostChart.jsx
import React from 'react';
import { DollarSign } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { AnalyticsService } from '../../services/analyticsService.js';

const CostChart = ({ summary }) => {
  if (!summary || !summary.cost)
 {
    return (
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 lg:col-span-2">
        <div className="flex items-center space-x-2 mb-6">
          <DollarSign className="w-6 h-6 text-yellow-400" />
          <h3 className="text-xl font-semibold text-gray-200">Cost Comparison</h3>
        </div>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No cost data available
        </div>
      </div>
    );
  }

  const costData = AnalyticsService.prepareCostData(summary);
  const metrics = AnalyticsService.calculateMetrics(summary);

  return (
    <div className="bg-gray-800 rounded-xl shadow-2xl p-6 lg:col-span-2 border-t-4 border-yellow-600">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <DollarSign className="w-6 h-6 text-yellow-400" />
          <h3 className="text-xl font-semibold text-gray-200">Cost Comparison</h3>
        </div>
        <div className="text-xs text-gray-400 bg-gray-700/50 px-3 py-1 rounded">
  Pricing: {summary.cost.model_pricing}
</div>

      </div>
      
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={costData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="name" 
              stroke="#9ca3af"
              fontSize={12}
            />
            <YAxis 
              stroke="#9ca3af"
              fontSize={12}
              tickFormatter={(value) => `$${value.toFixed(4)}`}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6'
              }}
              formatter={(value) => [`$${value.toFixed(4)}`, 'Cost']}
            />
            <Bar dataKey="cost" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div className="bg-gray-700/50 rounded-lg p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Total Savings</p>
          <p className="text-green-400 font-bold text-2xl">${metrics.totalSavings}</p>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-4 text-center">
          <p className="text-gray-400 text-sm mb-1">Cost Reduction</p>
          <p className="text-green-400 font-bold text-2xl">{metrics.savingsPercentage}%</p>
        </div>
      </div>
    </div>
  );
};

export default CostChart;