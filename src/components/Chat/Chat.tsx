import React, { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Bot, Sparkles, ArrowLeft } from 'lucide-react';
import { Message, ChatProps, SuggestionWithAnswer, ConversationContext } from '@/types/chat';
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { SuggestionButton } from './SuggestionButton';
import { generateResponse } from '@/utils/chatUtils';

const INITIAL_SYMPTOMS = [
    {
        question: "J'ai mal à la tête",
        answer: "Je comprends que vous avez mal à la tête. Je vais vous poser quelques questions pour mieux comprendre votre situation.",
        symptom: 'headache' as const
    },
];

export const Chat: React.FC<ChatProps> = ({ onClose }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [currentSuggestions, setCurrentSuggestions] = useState<SuggestionWithAnswer[]>([]);
    const [context, setContext] = useState<ConversationContext>({
        currentSymptom: 'unknown',
        initialMessage: true
    });
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
    const handleSendMessage = (content: string) => {
        // Add user's message
        addMessage(content, 'user');

        // Get response and suggestions based on current context
        const response = generateResponse(content, context);

        // Update context
        setContext(response.context);

        // Add bot's response after a delay
        setTimeout(() => {
            addMessage(response.response, 'bot');
            setCurrentSuggestions(response.suggestions);
        }, 1000);
    };

    const handleInitialSymptomClick = (symptom: typeof INITIAL_SYMPTOMS[0]) => {
        // Add user's selected symptom
        addMessage(symptom.question, 'user');

        // Add bot's initial response
        setTimeout(() => {
            addMessage(symptom.answer, 'bot');

            // Get and set follow-up suggestions with context
            const response = generateResponse(symptom.question, {
                currentSymptom: symptom.symptom,
                initialMessage: true
            });

            // Update context
            setContext(response.context);

            // Add follow-up message and suggestions
            addMessage(response.response, 'bot');
            setCurrentSuggestions(response.suggestions);
        }, 1000);
    };

    const handleSuggestionClick = (suggestion: SuggestionWithAnswer) => {
        // Don't send the question back to the bot - directly show the answer
        addMessage(suggestion.question, 'user');

        setTimeout(() => {
            addMessage(suggestion.answer, 'bot');

            // Get new suggestions maintaining the current context
            const response = generateResponse(suggestion.question, context);
            setContext(response.context);
            setCurrentSuggestions(response.suggestions);
        }, 1000);
    };

    return (
        <div className="fixed inset-0 bg-white z-50 md:p-4">
            <div className="h-full flex flex-col bg-white md:rounded-3xl md:shadow-2xl overflow-hidden">
                {/* Header */}
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

                {/* Chat Area */}
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
                                Décrivez vos symptômes ou sélectionnez une option ci-dessous.
                            </p>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
                                {INITIAL_SYMPTOMS.map((symptom, index) => (
                                    <SuggestionButton
                                        key={index}
                                        text={symptom.question}
                                        onClick={() => handleInitialSymptomClick(symptom)}
                                    />
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="max-w-3xl mx-auto space-y-6">
                            {messages.map((message) => (
                                <ChatMessage key={message.id} message={message} />
                            ))}

                            {/* Dynamic Suggestions */}
                            {currentSuggestions.length > 0 && (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                                    {currentSuggestions.map((suggestion, index) => (
                                        <SuggestionButton
                                            key={index}
                                            text={suggestion.question}
                                            onClick={() => handleSuggestionClick(suggestion)}
                                        />
                                    ))}
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="border-t border-gray-100 p-4 md:p-6 bg-white">
                    <div className="max-w-3xl mx-auto">
                        <ChatInput onSendMessage={handleSendMessage} />
                    </div>
                </div>
            </div>
        </div>
    );
};