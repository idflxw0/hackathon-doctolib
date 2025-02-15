import { Message } from '@/types/chat';
import { UserCircle, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isBot = message.sender === 'bot';

  return (
    <div className={`flex w-full mb-4 animate-fade-in ${isBot ? 'justify-start' : 'justify-end'}`}>
      <div className={`flex items-start gap-3 max-w-[80%] ${isBot ? 'flex-row' : 'flex-row-reverse'}`}>
        <div className="flex-shrink-0">
          {isBot ? (
            <div className="w-8 h-8 rounded-full bg-[#117acb] flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
          ) : (
            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
              <UserCircle className="w-5 h-5 text-gray-600" />
            </div>
          )}
        </div>

        <div className={`p-3 rounded-2xl ${isBot
          ? 'bg-white shadow-sm border border-gray-100 rounded-tl-none'
          : 'bg-[#117acb] text-white rounded-tr-none'
          }`}>
          <p className={`${isBot ? 'text-gray-900' : 'text-white'} text-base leading-relaxed`}>{message.content}</p>
          <span className={`text-xs mt-1 block ${isBot ? 'text-gray-400' : 'text-blue-100'
            }`}>
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  );
};
