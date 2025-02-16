import React from 'react';
import { ChevronRight } from 'lucide-react';
import { SuggestionButtonProps } from '@/types/chat';

export const SuggestionButton: React.FC<SuggestionButtonProps> = ({
    text,
    onClick,
    className = ''
}) => (
    <button
        onClick={onClick}
        className={`
      group p-4 text-left bg-white border border-gray-100 
      rounded-xl hover:border-blue-200 hover:shadow-sm 
      transition-all duration-200 w-full flex items-center justify-between
      ${className}
    `}
    >
        <span className="text-gray-900 group-hover:text-blue-600 transition-colors">
            {text}
        </span>
        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-600 transition-colors" />
    </button>
);