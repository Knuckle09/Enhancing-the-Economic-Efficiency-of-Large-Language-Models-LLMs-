// components/chat/MessageBubble.jsx
import React from "react";
import { User, Bot, BarChart3, Clock, Zap, Target, ThumbsDown, ThumbsUp } from "lucide-react";
import { MESSAGE_TYPES } from "../../data/constants.js";
import IconButton from "../common/IconButton.jsx";
import LoadingSpinner from "../common/LoadingSpinner.jsx";

const MessageBubble = ({ message, onAnalyzeMessage, isAnalyzing, onFeedback }) => {
  const isUser = message.type === MESSAGE_TYPES.USER;
  const isAssistant = message.type === MESSAGE_TYPES.ASSISTANT;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`flex space-x-3 max-w-4xl w-full ${
          isUser ? "flex-row-reverse space-x-reverse" : ""
        }`}
      >
        {/* Avatar */}
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
            isUser ? "bg-blue-600" : message.isError ? "bg-red-600" : "bg-gray-700"
          }`}
        >
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-gray-300" />
          )}
        </div>

        {/* Message Content */}
        <div
          className={`rounded-lg p-4 inline-block max-w-[70%] ${
            isUser
              ? "bg-blue-600 text-white ml-12"
              : message.isError
              ? "bg-red-900/50 text-red-200 mr-12 border border-red-700"
              : "bg-gray-800 text-gray-100 mr-12"
          } break-words whitespace-pre-wrap`}
        >
          {/* Model and Strategy info for assistant messages */}
          {isAssistant && !message.isError && message.model && (
            <div className="mb-3 pb-3 border-b border-gray-700 space-y-2">
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <Bot className="w-3 h-3" />
                <span>Model: <span className="text-blue-400 font-medium">{message.model}</span></span>
              </div>
              
              {message.strategyUsed && (
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <Target className="w-3 h-3" />
                  <span>Strategy: <span className="text-purple-400 font-medium capitalize">{message.strategyUsed}</span></span>
                </div>
              )}

              {message.category && (
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  <span>Category: <span className="text-green-400 font-medium capitalize">{message.category}</span></span>
                </div>
              )}
              {message.routingReason && (
                <div className="text-xs text-gray-400">Route: <span className="text-gray-300">{message.routingReason}</span></div>
              )}
              {message.cacheHit && (
                <div className="text-xs text-cyan-300">Reused from the recent similarity cache</div>
              )}
            </div>
          )}

          {/* Message text */}
          <div className="text-sm leading-relaxed">{message.content}</div>
          {isAssistant && message.isStreaming && (
            <LoadingSpinner className="mt-3" text={message.content ? (message.streamStatus || 'Generating response…') : (message.streamStatus || 'Connecting…')} />
          )}

          {/* Metrics for assistant messages */}
          {isAssistant && !message.isError && (message.tokenReductionPercent !== undefined || message.similarity !== undefined) && (
            <div className="mt-3 pt-3 border-t border-gray-700 grid grid-cols-2 gap-3">
              {message.tokenReductionPercent !== undefined && (
                <div className="flex items-center gap-2 text-xs">
                  <Zap className="w-3 h-3 text-yellow-400" />
                  <span className="text-gray-400">
                    Token Reduction: <span className="text-yellow-400 font-semibold">{message.tokenReductionPercent.toFixed(1)}%</span>
                  </span>
                </div>
              )}
              
              {message.similarity !== undefined && (
                <div className="flex items-center gap-2 text-xs">
                  <Target className="w-3 h-3 text-green-400" />
                  <span className="text-gray-400">
                    Semantic similarity: <span className="text-green-400 font-semibold">{Number.isFinite(message.similarity) ? `${(message.similarity * 100).toFixed(1)}%` : 'Not measured'}</span>
                  </span>
                </div>
              )}

              {message.similarityMethod && (
                <div className="text-xs text-gray-500 col-span-2">Similarity method: {message.similarityMethod}</div>
              )}

              {message.processingTime !== undefined && (
                <div className="flex items-center gap-2 text-xs col-span-2">
                  <Clock className="w-3 h-3 text-blue-400" />
                  <span className="text-gray-400">
                    Processing Time: <span className="text-blue-400 font-semibold">{message.processingTime.toFixed(2)}s</span>
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Analyze button for assistant messages with API data */}
          {isAssistant && !message.isError && message.fullApiResponse && (
            <div className="mt-4 pt-3 border-t border-gray-700">
              <IconButton
                icon={isAnalyzing ? LoadingSpinner : BarChart3}
                onClick={() => onAnalyzeMessage(message)}
                disabled={isAnalyzing}
                text={isAnalyzing ? "Analyzing..." : "View Analytics"}
                variant="analytics"
                size="sm"
              />
            </div>
          )}

          {isAssistant && !message.isError && message.fullApiResponse && onFeedback && (
            <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
              <span>{message.feedback ? 'Thanks for the feedback' : 'Was this response useful?'}</span>
              <button aria-label="Useful response" title="Useful response" disabled={Boolean(message.feedback)}
                onClick={() => onFeedback(message, 1)} className={`rounded p-1 hover:bg-gray-700 disabled:opacity-70 ${message.feedback === 1 ? 'text-green-400' : 'text-gray-400'}`}>
                <ThumbsUp className="w-4 h-4" />
              </button>
              <button aria-label="Not useful response" title="Not useful response" disabled={Boolean(message.feedback)}
                onClick={() => onFeedback(message, -1)} className={`rounded p-1 hover:bg-gray-700 disabled:opacity-70 ${message.feedback === -1 ? 'text-red-400' : 'text-gray-400'}`}>
                <ThumbsDown className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
