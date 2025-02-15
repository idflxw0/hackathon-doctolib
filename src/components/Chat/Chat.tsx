import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Message } from '@/types/chat';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { MessageCircle } from 'lucide-react';

export const Chat = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = (content: string) => {
        const newMessage: Message = {
            id: uuidv4(),
            content,
            sender: 'user',
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, newMessage]);

        // Simulate bot response
        setTimeout(() => {
            const botMessage: Message = {
                id: uuidv4(),
                content: "I'm here to help you. How can I assist you today?",
                sender: 'bot',
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, botMessage]);
        }, 1000);
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
                <div className="bg-[#117acb] p-4">
                    <div className="flex items-center gap-3">
                        <MessageCircle className="text-white w-6 h-6" />
                        <h1 className="text-white text-xl font-semibold">
                            AI Assistant
                        </h1>
                    </div>
                </div>

                <div className="h-96 md:h-[32rem] overflow-y-auto p-4 bg-gray-50">
                    {messages.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full text-gray-500">
                            <MessageCircle className="w-12 h-12 mb-4 opacity-50" />
                            <p className="text-lg">Start a conversation</p>
                            <p className="text-sm">Type a message to begin chatting</p>
                        </div>
                    ) : (
                        messages.map((message) => (
                            <ChatMessage key={message.id} message={message} />
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </div>

                <div className="border-t border-gray-100 p-4 bg-white">
                    <ChatInput onSendMessage={handleSendMessage} />
                </div>
            </div>
        </div>
    );
};
