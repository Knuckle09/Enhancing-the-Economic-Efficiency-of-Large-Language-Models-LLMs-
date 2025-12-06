// components/chat/ChatInterface.jsx
// Main chat interface component - purely UI focused

import React from 'react';
import { useChat } from '../../hooks/useChat.js';
import { useAnalytics } from '../../hooks/useAnalytics.js';
import ChatHeader from './ChatHeader.jsx';
import ChatMessages from './ChatMessages.jsx';
import ChatInput from './ChatInput.jsx';

const ChatInterface = () => {
  const { messages, isLoading } = useChat();
  const { analyzeMessage, isAnalyzing } = useAnalytics();

  const handleAnalyzeMessage = (message) => {
    analyzeMessage(message);
  };

  return (
    <div className="h-screen bg-gray-900 text-gray-100 flex flex-col">
      <ChatHeader />
      <ChatMessages 
        messages={messages}
        isLoading={isLoading}
        onAnalyzeMessage={handleAnalyzeMessage}
        isAnalyzing={isAnalyzing}
      />
      <ChatInput />
    </div>
  );
};

export default ChatInterface;