// components/analytics/PromptDetailsCard.jsx
import React from 'react';
import { ChevronDown, ChevronUp, Zap, Target, Clock, Cpu } from 'lucide-react';

const PromptDetailsCard = ({ prompt, isExpanded, onToggle }) => {
  if (!prompt) return null;

  return (
    <div className="bg-gray-800 rounded-xl shadow-2xl overflow-hidden border border-gray-700">
      {/* Header */}
      <div 
        className="p-6 cursor-pointer hover:bg-gray-750 transition-colors"
        onClick={() => onToggle(prompt.id)}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className="px-3 py-1 bg-blue-900/50 text-blue-300 text-xs rounded-full font-medium capitalize">
                {prompt.category || prompt.type}
              </span>
              {prompt.strategy_used && (
                <span className="px-3 py-1 bg-purple-900/50 text-purple-300 text-xs rounded-full font-medium capitalize">
                  {prompt.strategy_used} Strategy
                </span>
              )}
            </div>
            <h4 className="text-lg font-semibold text-gray-200 mb-2">
              {prompt.original_prompt}
            </h4>
            
            {/* Metrics Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <div>
                  <p className="text-xs text-gray-400">Token Reduction</p>
                  <p className="text-sm font-semibold text-yellow-400">
                    {prompt.metrics?.token_reduction_percent?.toFixed(1) || 0}%
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-green-400" />
                <div>
                  <p className="text-xs text-gray-400">Similarity</p>
                  <p className="text-sm font-semibold text-green-400">
                    {prompt.metrics?.token_reduction_percent?.toFixed(1) || 0}%
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-blue-400" />
                <div>
                  <p className="text-xs text-gray-400">Tokens Saved</p>
                  <p className="text-sm font-semibold text-blue-400">
                    {prompt.metrics?.tokens_saved || 0}
                  </p>
                </div>
              </div>
              
              {prompt.processing_time && (
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-purple-400" />
                  <div>
                    <p className="text-xs text-gray-400">Processing</p>
                    <p className="text-sm font-semibold text-purple-400">
                      {prompt.processing_time.toFixed(2)}s
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <button className="ml-4 text-gray-400 hover:text-gray-200 transition-colors">
            {isExpanded ? (
              <ChevronUp className="w-6 h-6" />
            ) : (
              <ChevronDown className="w-6 h-6" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-gray-700 p-6 space-y-6 bg-gray-900/50">
          {/* Original Prompt */}
          <div>
            <h5 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full"></span>
              Original Prompt
            </h5>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <p className="text-gray-300 text-sm leading-relaxed">
                {prompt.original_prompt}
              </p>
            </div>
          </div>

          {/* Optimized Prompt */}
          <div>
            <h5 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span>
              Optimized Prompt
            </h5>
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <p className="text-gray-300 text-sm leading-relaxed">
                {prompt.optimized_prompt}
              </p>
            </div>
          </div>

          {/* LLM Response */}
          {prompt.llm_response && (
            <div>
              <h5 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
                <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                LLM Response
                {prompt.selected_llm && (
                  <span className="text-xs text-gray-500">({prompt.selected_llm})</span>
                )}
              </h5>
              <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 max-h-96 overflow-y-auto">
                <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">
                  {prompt.llm_response}
                </p>
              </div>
            </div>
          )}

          {/* Token Details */}
          {prompt.metrics && (
            <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-700">
              <div className="text-center bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-400 mb-1">Original Tokens</p>
                <p className="text-lg font-bold text-red-400">
                  {prompt.metrics.original_tokens || 0}
                </p>
              </div>
              <div className="text-center bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-400 mb-1">Optimized Tokens</p>
                <p className="text-lg font-bold text-green-400">
                  {prompt.metrics.optimized_tokens || 0}
                </p>
              </div>
              <div className="text-center bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-400 mb-1">Tokens Saved</p>
                <p className="text-lg font-bold text-blue-400">
                  {prompt.metrics.tokens_saved || 0}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PromptDetailsCard;