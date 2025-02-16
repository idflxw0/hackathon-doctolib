import React from 'react';
import { Bot } from 'lucide-react';
import { ChatMessageProps } from '@/types/chat';

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isBot = message.sender === 'bot';

  return (
    <div className={`flex w-full animate-fade-in ${isBot ? 'justify-start' : 'justify-end'
      }`}>
      <div className={`flex items-start gap-4 max-w-[85%] ${isBot ? 'flex-row' : 'flex-row-reverse'
        }`}>
        <div className="flex-shrink-0">
          {isBot ? (
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-blue-700 flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
          ) : (
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
              <div className="w-6 h-6 rounded-full bg-gray-300" />
            </div>
          )}
        </div>
        <div className={`p-4 rounded-2xl ${isBot
          ? 'bg-white border border-gray-100 rounded-tl-none'
          : 'bg-blue-600 text-white rounded-tr-none'
          }`}>
          <p className="text-base leading-relaxed">{message.content}</p>
          <span className="text-xs mt-2 block opacity-60">
            {message.timestamp.toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            })}
          </span>
        </div>
      </div>
    </div>
  );
};