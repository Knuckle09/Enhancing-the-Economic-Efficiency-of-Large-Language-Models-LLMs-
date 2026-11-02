import React, { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext.jsx';
import apiService from '../../services/api.js';
import IconButton from '../common/IconButton.jsx';

const showOllamaModels = import.meta.env.VITE_ENABLE_OLLAMA_MODELS === 'true';
const fallbackModels = [
  { id: 'gemini-pro', name: 'Gemini 2.5 Pro', provider: 'gemini' },
  { id: 'gemini-flash', name: 'Gemini 2.5 Flash', provider: 'gemini' },
  { id: 'qwen-coder', name: 'Qwen 2.5 Coder', provider: 'ollama' },
  { id: 'qwen-math', name: 'Qwen Math', provider: 'ollama' },
  { id: 'tinyllama', name: 'TinyLlama', provider: 'ollama' },
  { id: 'phi-3', name: 'Phi-3', provider: 'ollama' },
  { id: 'codellama', name: 'CodeLlama', provider: 'ollama' },
];

const ChatInput = () => {
  const {
    currentPrompt, setCurrentPrompt, sendMessage, isLoading,
    modelPreference, setModelPreference, selectedModel, setSelectedModel,
    selectedStrategy, setStrategy,
  } = useChat();
  const [models, setModels] = useState(fallbackModels);
  const textareaRef = useRef(null);

  useEffect(() => {
    let active = true;
    apiService.get('/api/models').then((response) => {
      if (active && Array.isArray(response.models)) setModels(response.models);
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  const cloudModels = models.filter((model) => model.provider === 'gemini');
  const localModels = models.filter((model) => model.provider === 'ollama');
  const handleSendMessage = () => {
    if (currentPrompt.trim()) sendMessage(currentPrompt);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 44), 200)}px`;
    textarea.style.overflowY = textarea.scrollHeight > 200 ? 'auto' : 'hidden';
  }, [currentPrompt]);

  return (
    <div className="sticky bottom-0 bg-gray-900/80 backdrop-blur-md p-4 z-10">
      <div className="max-w-6xl mx-auto space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex bg-gray-800 rounded-lg p-1">
            {['auto', 'manual'].map((mode) => (
              <button key={mode} onClick={() => setModelPreference(mode)} disabled={isLoading}
                className={`px-4 py-1.5 text-sm rounded-md transition-all ${modelPreference === mode ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}>
                {mode === 'auto' ? 'Auto' : 'Manual'}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="optimization-strategy">Prompt optimization strategy</label>
            <select id="optimization-strategy" value={selectedStrategy} onChange={(event) => setStrategy(event.target.value)} disabled={isLoading}
              className="bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-2 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="conservative">Conservative optimization</option>
              <option value="balanced">Balanced optimization</option>
              <option value="aggressive">Aggressive optimization</option>
            </select>

            {modelPreference === 'manual' && (
              <select value={selectedModel || ''} onChange={(event) => setSelectedModel(event.target.value)} disabled={isLoading}
                className="bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-2 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="" disabled>Select a model</option>
                {cloudModels.length > 0 && <optgroup label="Cloud models">
                  {cloudModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                </optgroup>}
                {showOllamaModels && localModels.length > 0 && <optgroup label="Local Ollama models">
                  {localModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                </optgroup>}
                {cloudModels.length === 0 && (!showOllamaModels || localModels.length === 0) && <option value="" disabled>No models available from this backend</option>}
              </select>
            )}
          </div>
        </div>

        <div className="flex items-end gap-3 bg-gray-700/50 border border-gray-600 rounded-2xl p-3">
          <textarea ref={textareaRef} value={currentPrompt} onChange={(event) => setCurrentPrompt(event.target.value)}
            onKeyDown={handleKeyPress} placeholder={modelPreference === 'manual' ? 'Type your message… (manual model)' : 'Type your message…'}
            className="flex-1 bg-transparent text-gray-100 placeholder-gray-400 focus:outline-none resize-none overflow-y-hidden min-h-[20px] leading-5"
            rows={1} style={{ maxHeight: '200px', minHeight: '20px' }} disabled={isLoading} />
          <IconButton icon={Send} onClick={handleSendMessage}
            disabled={isLoading || !currentPrompt.trim() || (modelPreference === 'manual' && !selectedModel)}
            variant="primary" size="md" />
        </div>
        <div className="text-xs text-gray-400 flex justify-between gap-3">
          <span>Enter to send · Shift+Enter for a new line · recent chat turns are kept in this browser</span>
          {modelPreference === 'manual' && !selectedModel && <span className="text-yellow-400">Select an available model</span>}
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
