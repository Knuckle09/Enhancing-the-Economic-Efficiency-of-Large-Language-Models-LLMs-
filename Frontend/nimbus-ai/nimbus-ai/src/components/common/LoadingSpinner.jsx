// components/common/LoadingSpinner.jsx
import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = ({ size = 'w-4 h-4', text = null, className = '' }) => {
  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <Loader2 className={`${size} animate-spin text-blue-400`} />
      {text && <span className="text-gray-300">{text}</span>}
    </div>
  );
};

export default LoadingSpinner;