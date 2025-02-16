import React, { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Bot, Sparkles, ArrowLeft } from 'lucide-react';
import { Message, ChatProps } from '@/types/chat';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';

export const Chat: React.FC<ChatProps> = ({ onClose }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const addMessage = (content: string, sender: 'user' | 'bot') => {
        const newMessage: Message = {
            id: uuidv4(),
            content,
            sender,
            timestamp: new Date(),
        };
        setMessages(prev => [...prev, newMessage]);
    };

    const handleSendMessage = async (content: string) => {
        addMessage(content, 'user');

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: content }),
            });

            const data = await response.json();
            addMessage(data.reply, 'bot');
        } catch (error) {
            console.error('Erreur API:', error);
            addMessage("Je rencontre un problème. Réessayez plus tard.", 'bot');
        }
    };

    return (
        <div className="fixed inset-0 bg-white z-50 md:p-4">
            <div className="h-full flex flex-col bg-white md:rounded-3xl md:shadow-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-blue-700 p-4 md:p-6">
                    <div className="flex items-center justify-between mb-6">
                        <button
                            onClick={onClose}
                            className="text-white hover:bg-white/10 p-2 rounded-full transition-colors"
                        >
                            <ArrowLeft className="w-6 h-6" />
                        </button>
                        <h1 className="text-xl font-semibold text-white">Assistant Santé</h1>
                        <div className="w-10" />
                    </div>

                    <div className="flex items-center gap-3 text-white/90">
                        <div className="bg-white/10 p-2 rounded-full">
                            <Bot className="w-5 h-5" />
                        </div>
                        <div>
                            <p className="font-medium">Assistant Médical</p>
                            <p className="text-sm text-white/70">Disponible 24/7</p>
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto px-4 py-6 bg-gray-50">
                    {messages.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-center px-4">
                            <div className="bg-blue-50 p-4 rounded-full mb-6">
                                <Sparkles className="w-8 h-8 text-blue-600" />
                            </div>
                            <h2 className="text-xl font-semibold text-gray-900 mb-2">
                                Comment puis-je vous aider aujourd'hui ?
                            </h2>
                            <p className="text-gray-500 mb-8 max-w-md">
                                Décrivez vos symptômes.
                            </p>
                        </div>
                    ) : (
                        <div className="max-w-3xl mx-auto space-y-6">
                            {messages.map((message) => (
                                <ChatMessage key={message.id} message={message} />
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                <div className="border-t border-gray-100 p-4 md:p-6 bg-white">
                    <div className="max-w-3xl mx-auto">
                        <ChatInput onSendMessage={handleSendMessage} />
                    </div>
                </div>
            </div>
        </div>
    );
};
