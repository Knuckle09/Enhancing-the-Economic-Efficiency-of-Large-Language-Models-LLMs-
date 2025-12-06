// components/common/IconButton.jsx
import React from 'react';

const IconButton = ({ 
  icon: Icon, 
  onClick, 
  disabled = false, 
  text, 
  variant = 'primary',
  size = 'md',
  className = '' 
}) => {
  const baseStyles = 'font-medium rounded-lg transition-all duration-200 flex items-center justify-center space-x-2 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white',
    secondary: 'bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white border border-gray-700',
    analytics: 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-700 text-white'
  };

  const sizes = {
    sm: 'py-1 px-2 text-xs',
    md: 'py-2 px-4 text-sm',
    lg: 'py-3 px-6 text-base'
  };

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
    >
      <Icon className={iconSizes[size]} />
      {text && <span>{text}</span>}
    </button>
  );
};

export default IconButton;