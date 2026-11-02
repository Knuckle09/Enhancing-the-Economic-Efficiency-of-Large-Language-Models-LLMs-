import React from 'react';
import { Cpu } from 'lucide-react';

const ModelPerformanceChart = ({ prompts = [] }) => {
  const grouped = prompts.reduce((result, prompt) => {
    const model = prompt.selected_llm || 'Unknown';
    result[model] ||= { model, replies: 0, time: 0, rated: 0, positive: 0 };
    const entry = result[model];
    entry.replies += 1;
    entry.time += Number(prompt.processing_time || 0);
    if (prompt.feedback) {
      entry.rated += 1;
      if (prompt.feedback === 1) entry.positive += 1;
    }
    return result;
  }, {});
  const rows = Object.values(grouped).sort((a, b) => b.replies - a.replies);

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
      <div className="flex items-center gap-2 mb-2">
        <Cpu className="w-5 h-5 text-blue-400" />
        <h3 className="text-lg font-semibold text-gray-200">Model Performance</h3>
      </div>
      <p className="text-xs text-gray-400 mb-4">Real response times and optional user ratings</p>
      {rows.length ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase text-gray-400 border-b border-gray-700">
              <tr><th className="py-3 pr-3">Model</th><th className="py-3 px-3">Replies</th><th className="py-3 px-3">Avg. time</th><th className="py-3 pl-3">Useful ratings</th></tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model} className="border-b border-gray-700/60 last:border-0">
                  <td className="py-3 pr-3 text-blue-300 font-medium">{row.model}</td>
                  <td className="py-3 px-3 text-gray-200">{row.replies}</td>
                  <td className="py-3 px-3 text-gray-200">{(row.time / row.replies).toFixed(2)}s</td>
                  <td className="py-3 pl-3 text-gray-200">{row.rated ? `${row.positive}/${row.rated}` : 'Not rated'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div className="h-48 flex items-center justify-center text-gray-500">No model performance data yet</div>}
    </div>
  );
};

export default ModelPerformanceChart;
