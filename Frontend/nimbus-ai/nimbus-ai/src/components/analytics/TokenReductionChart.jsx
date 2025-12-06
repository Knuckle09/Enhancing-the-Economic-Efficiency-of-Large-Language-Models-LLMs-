import React from 'react';
import { TrendingDown } from 'lucide-react';

const TokenReductionChart = ({ data, summary }) => {
  // Support both 'data' and 'summary' props for flexibility
  const sourceData = summary || data;
  
  // Fixed: Check ALL possible paths where token_reduction_percent might exist
  const tokenReduction = Number(
    sourceData?.token_reduction_percent ??                    // Direct on data
    sourceData?.data?.token_reduction_percent ??              // Nested in data.data
    sourceData?.avg_token_reduction ??                        // Summary field
    sourceData?.tokens?.token_reduction_percent ??            // Inside tokens object
    sourceData?.summary?.token_reduction_percent ??           // Inside summary
    sourceData?.summary?.avg_token_reduction ??               // Summary average
    0
  );
  const remaining = 100 - tokenReduction;

  const getColor = (percent) => {
    if (percent >= 30) return '#10b981';
    if (percent >= 20) return '#3b82f6';
    if (percent >= 10) return '#f59e0b';
    return '#ef4444';
  };

  const color = getColor(tokenReduction);

  const circleStyle = {
    background: `conic-gradient(
      ${color} 0% ${tokenReduction}%,
      #374151 ${tokenReduction}% 100%
    )`,
  };

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
      <div className="flex items-center space-x-2 mb-6">
        <TrendingDown className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold text-gray-200">Token Reduction</h3>
      </div>

      <div className="flex justify-center mb-6">
        <div className="w-48 h-48 rounded-full p-2" style={circleStyle}>
          <div className="w-full h-full rounded-full bg-gray-900 flex items-center justify-center">
            <div className="text-center">
              <div className="flex items-baseline justify-center mb-1">
                <span className="text-5xl font-bold" style={{ color }}>
                  {tokenReduction.toFixed(1)}
                </span>
                <span className="text-2xl font-semibold text-gray-400">%</span>
              </div>
              <span className="text-xs text-gray-400">out of 100%</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: color }} />
          <div>
            <p className="text-xs text-gray-400">Tokens Reduced</p>
            <p className="text-lg font-bold" style={{ color }}>
              {tokenReduction.toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 rounded-full bg-gray-500" />
          <div>
            <p className="text-xs text-gray-400">Tokens Retained</p>
            <p className="text-lg font-bold text-gray-400">{remaining.toFixed(1)}%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TokenReductionChart;