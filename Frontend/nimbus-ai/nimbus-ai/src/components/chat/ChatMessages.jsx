// components/chat/ChatMessages.jsx
import React, { useEffect, useRef } from 'react';
import { Bot } from 'lucide-react';
import MessageBubble from './MessageBubble.jsx';
import LoadingSpinner from '../common/LoadingSpinner.jsx';

const ChatMessages = ({ messages, isLoading, onAnalyzeMessage, isAnalyzing }) => {
  const messagesEndRef = useRef(null);

  // Scroll to the bottom whenever messages or loading state changes
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto p-4 min-h-0">
      <div className="max-w-6xl mx-auto space-y-6 pb-4">
        {/* Welcome message when no chat history */}
        {messages.length === 0 && (
          <div className="text-center py-16">
            <Bot className="w-20 h-20 text-gray-500 mx-auto mb-6" />
            <h2 className="text-3xl md:text-4xl font-bold text-gray-200 mb-4">
              Welcome to Nimbus AI
            </h2>
            <p className="text-lg md:text-xl text-gray-400">
              Start a conversation by typing your message below
            </p>
          </div>
        )}

        {/* Message bubbles */}
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onAnalyzeMessage={onAnalyzeMessage}
            isAnalyzing={isAnalyzing}
          />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex space-x-3 max-w-4xl w-full">
              <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-gray-300" />
              </div>
              <div className="bg-gray-800 rounded-lg p-4 mr-12">
                <LoadingSpinner text="Thinking..." />
              </div>
            </div>
          </div>
        )}

        {/* Dummy div to scroll into view */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default ChatMessages;