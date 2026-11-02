// components/chat/ChatHeader.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { Info, Trash2 } from 'lucide-react';
import Logo from '../../assets/logo.jpg'; // adjust path if needed
import { useChat } from '../../contexts/ChatContext.jsx';

const ChatHeader = () => {
  const { messages, clearMessages } = useChat();
  return (
    <div className="bg-gray-800 border-b border-gray-700 p-4 flex-shrink-0">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        
        {/* Left: Logo + Title */}
        <div className="flex items-center gap-3">
          <img 
            src={Logo} 
            alt="Nimbus AI Logo" 
            className="w-15 h-10 rounded-md object-cover"
          />
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Nimbus AI Chat
            </h1>
            <p className="text-gray-400 text-sm">
              Powered by multiple LLM models with advanced analytics
            </p>
          </div>
        </div>

        {/* Right: About Link */}
        <div className="flex items-center gap-4">
          {messages.length > 0 && (
            <button onClick={clearMessages} className="text-gray-400 hover:text-white flex items-center gap-2 text-sm" title="Clear saved chat">
              <Trash2 className="w-4 h-4" /> <span className="hidden sm:inline">Clear chat</span>
            </button>
          )}
          <Link to="/about" className="text-gray-400 hover:text-white">
            <Info className="w-6 h-6" title="About" />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ChatHeader;
