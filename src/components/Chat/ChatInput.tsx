import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { ChatInputProps } from '@/types/chat';

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage }) => {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    setLoading(true);
    await onSendMessage(message);
    setMessage('');
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Décrivez vos symptômes..."
        className="flex-1 p-4 rounded-2xl bg-gray-50 focus:bg-white border border-gray-100 focus:border-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all duration-200"
      />
      <button
        type="submit"
        disabled={!message.trim() || loading}
        className="bg-blue-600 text-white p-4 rounded-2xl hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? '...' : <Send className="w-5 h-5" />}
      </button>
    </form>
  );
};
