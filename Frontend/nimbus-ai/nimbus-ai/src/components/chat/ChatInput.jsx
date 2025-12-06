import React, { useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext.jsx';
import IconButton from '../common/IconButton.jsx';

const ChatInput = () => {
  const {
  currentPrompt,
  setCurrentPrompt,
  sendMessage,
  isLoading,
  modelPreference,
  setModelPreference,
  selectedModel,
  setSelectedModel,
} = useChat();

  const textareaRef = useRef(null);

  const handleSendMessage = () => {
    if (currentPrompt.trim()) {
      sendMessage(currentPrompt);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Auto-resize logic
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = 'auto';
      
      // Calculate the new height (minimum 1 line, maximum ~8 lines at 200px)
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 44), 200);
      textarea.style.height = newHeight + 'px';
      
      // Show scroll only when content exceeds max height
      if (textarea.scrollHeight > 200) {
        textarea.style.overflowY = 'auto';
      } else {
        textarea.style.overflowY = 'hidden';
      }
    }
  }, [currentPrompt]);

  return (
  <div className="sticky bottom-0 bg-gray-900/80 backdrop-blur-md p-4 z-10">
    <div className="max-w-6xl mx-auto space-y-2">

      {/* 🔹 MODEL CONTROL BAR */}
      <div className="flex items-center justify-between gap-3">
        
        {/* Auto / Manual segmented control */}
        <div className="flex bg-gray-800 rounded-lg p-1">
          {['auto', 'manual'].map((mode) => (
            <button
              key={mode}
              onClick={() => setModelPreference(mode)}
              disabled={isLoading}
              className={`
                px-4 py-1.5 text-sm rounded-md transition-all
                ${modelPreference === mode
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-gray-400 hover:text-gray-200'
                }
              `}
            >
              {mode === 'auto' ? 'Auto' : 'Manual'}
            </button>
          ))}
        </div>

        {/* Model dropdown (Manual only) */}
        {modelPreference === 'manual' && (
          <select
            value={selectedModel || ''}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isLoading}
            className="
              bg-gray-800 text-gray-200 text-sm
              rounded-lg px-3 py-2
              border border-gray-700
              focus:outline-none focus:ring-2 focus:ring-blue-500
            "
          >
            <option value="" disabled>
              Select a model
            </option>

            <optgroup label="🌐 Cloud Models">
              <option value="gemini-pro">Gemini Pro – best for complex tasks</option>
              <option value="gemini-flash">Gemini Flash – fast & efficient</option>
            </optgroup>

            <optgroup label="💻 Local Models">
              <option value="codellama">CodeLlama – coding specialist</option>
              <option value="qwen-math">Qwen Math – math specialist</option>
              <option value="tinyllama">TinyLlama – fast general purpose</option>
              <option value="phi-3">Phi-3 – efficient model</option>
            </optgroup>
          </select>
        )}
      </div>

      {/* 🔹 CHAT INPUT */}
      <div className="flex items-end gap-3 bg-gray-700/50 border border-gray-600 rounded-2xl p-3">
        <textarea
          ref={textareaRef}
          value={currentPrompt}
          onChange={(e) => setCurrentPrompt(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={
            modelPreference === 'manual'
              ? 'Type your message… (manual model)'
              : 'Type your message…'
          }
          className="
            flex-1 bg-transparent
            text-gray-100 placeholder-gray-400
            focus:outline-none resize-none
            overflow-y-hidden min-h-[20px] leading-5
          "
          rows={1}
          style={{ maxHeight: '200px', minHeight: '20px' }}
          disabled={isLoading}
        />

        <IconButton
          icon={Send}
          onClick={handleSendMessage}
          disabled={
            isLoading ||
            !currentPrompt.trim() ||
            (modelPreference === 'manual' && !selectedModel)
          }
          variant="primary"
          size="md"
        />
      </div>

      {/* 🔸 Helper text */}
      <div className="text-xs text-gray-400 flex justify-between">
        <span>Press Enter to send • Shift+Enter for new line</span>
        {modelPreference === 'manual' && !selectedModel && (
          <span className="text-yellow-400">
            Select a model to continue
          </span>
        )}
      </div>

    </div>
  </div>
);

};

export default ChatInput;