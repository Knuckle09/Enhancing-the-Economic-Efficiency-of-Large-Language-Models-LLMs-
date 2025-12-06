// components/analytics/ProcessingMetricsCard.jsx
import React from 'react';
import { Activity, Target, Clock, CheckCircle } from 'lucide-react';

const ProcessingMetricsCard = ({ summary }) => {
  if (!summary) return null;

  // Extract metrics from API summary
  const metrics = [
    {
      label: 'LLM Used',
      value: summary.selected_llm || 'N/A',
      icon: <Activity className="w-5 h-5 text-purple-400" />,
      color: 'text-purple-400'
    },
    {
      label: 'Category',
      value: summary.category || 'N/A',
      icon: <Target className="w-5 h-5 text-blue-400" />,
      color: 'text-blue-400'
    },
    {
      label: 'Processing Time',
      value: summary.processing_time ? `${summary.processing_time}s` : 'N/A',
      icon: <Clock className="w-5 h-5 text-green-400" />,
      color: 'text-green-400'
    },
    {
      label: 'Status',
      value: summary.target_achieved ? 'Success' : 'Partial',
      icon: summary.target_achieved ? 
        <CheckCircle className="w-5 h-5 text-green-400" /> :
        <Activity className="w-5 h-5 text-yellow-400" />,
      color: summary.target_achieved ? 'text-green-400' : 'text-yellow-400'
    }
  ];

  return (
    <div className="bg-gray-800 rounded-xl shadow-2xl p-6 mb-8">
      <div className="flex items-center space-x-2 mb-6">
        <Activity className="w-6 h-6 text-indigo-400" />
        <h3 className="text-xl font-semibold text-gray-200">Processing Metrics</h3>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        {metrics.map((metric, idx) => (
          <div key={idx} className="bg-gray-700/50 rounded-lg p-4 hover:bg-gray-700 transition-colors">
            <div className="flex items-center space-x-2 mb-2">
              {metric.icon}
              <span className="text-gray-400 text-sm">{metric.label}</span>
            </div>
            <p className={`${metric.color} font-bold text-lg`}>{metric.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessingMetricsCard;
