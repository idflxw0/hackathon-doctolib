export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface SuggestionWithAnswer {
  question: string;
  answer: string;
}

export interface ConversationContext {
  currentSymptom: 'headache' | 'fever' | 'cough' | 'unknown';
  initialMessage: boolean;
}

export interface BotResponse {
  response: string;
  suggestions: SuggestionWithAnswer[];
  context: ConversationContext;
}

export interface SuggestionButtonProps {
  text: string;
  onClick: () => void;
  className?: string;
}

export interface ChatInputProps {
  onSendMessage: (message: string) => void;
}

export interface ChatMessageProps {
  message: Message;
}

export interface ChatProps {
  onClose: () => void;
}

export interface ChatHistory {
  messages: Message[];
} 