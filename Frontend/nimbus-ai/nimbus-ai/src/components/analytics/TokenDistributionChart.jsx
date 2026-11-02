import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const COLORS = ['#3b82f6', '#10b981'];

const TokenDistributionChart = ({ data }) => {
  const used = Number(data?.tokens?.optimized_tokens || 0);
  const saved = Number(data?.tokens?.tokens_saved || 0);
  const distribution = [
    { name: 'Tokens sent after optimization', value: used },
    { name: 'Estimated tokens saved', value: saved },
  ].filter((item) => item.value > 0);

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
      <h3 className="text-lg font-semibold text-gray-200 mb-2">Token Usage</h3>
      <p className="text-xs text-gray-400 mb-3">Approximate counts from your saved replies</p>
      {distribution.length ? (
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={distribution} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={3}>
                {distribution.map((entry, index) => <Cell key={entry.name} fill={COLORS[index]} />)}
              </Pie>
              <Tooltip formatter={(value) => [`${value} estimated tokens`, '']} contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      ) : <div className="h-64 flex items-center justify-center text-gray-500">No token usage data yet</div>}
    </div>
  );
};

export default TokenDistributionChart;
