import React from 'react';
import { Activity, DollarSign, MessageSquare, Zap } from 'lucide-react';
import { REPORT_BENCHMARKS } from '../../data/constants.js';

const ReportBenchmarksCard = () => {
  const { costSavingsRangePercent, semanticSimilarityMinimum, semanticSimilarityReported } = REPORT_BENCHMARKS;
  const metrics = [
    { label: 'Token reduction', value: `${REPORT_BENCHMARKS.tokenReductionPercent}%`, icon: Zap, color: 'text-blue-300' },
    { label: 'Cost savings', value: `${costSavingsRangePercent[0]}–${costSavingsRangePercent[1]}%`, icon: DollarSign, color: 'text-yellow-300' },
    { label: 'Semantic similarity', value: `>${(semanticSimilarityMinimum * 100).toFixed(0)}% · ${(semanticSimilarityReported * 100).toFixed(0)}% reported`, icon: Activity, color: 'text-green-300' },
    { label: 'Multi-turn coherence', value: `+${REPORT_BENCHMARKS.multiTurnCoherenceImprovementPercent}%`, icon: MessageSquare, color: 'text-purple-300' },
  ];

  return (
    <section className="bg-gray-800/70 rounded-xl border border-indigo-500/30 p-5 mb-8">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
        <div>
          <h2 className="text-xl font-semibold text-gray-100">G2 Report Benchmarks</h2>
          <p className="text-sm text-gray-400 mt-1">Published study results; shown separately from live session measurements.</p>
        </div>
        <span className="text-xs text-indigo-200 bg-indigo-900/50 rounded-full px-3 py-1">
          {REPORT_BENCHMARKS.singleTurnPrompts} single-turn prompts · {REPORT_BENCHMARKS.multiTurnConversations} multi-turn conversations
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        {metrics.map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="bg-gray-900/70 rounded-lg p-4">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
              <Icon className={`w-4 h-4 ${color}`} />
              <span>{label}</span>
            </div>
            <p className={`text-lg font-semibold ${color}`}>{value}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default ReportBenchmarksCard;
