"use client";

import React from 'react';
import { useChat } from '@/components/chat/useChat';
import { ChatMessages } from '@/components/chat/ChatMessages';
import { EnhancedChatInput } from '@/components/chat/EnhancedChatInput';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { MessageCircle, Trash2 } from 'lucide-react';

export function AiChatView() {
  const {
    messages,
    inputText,
    isLoading,
    error,
    referencedMedia,
    handleInputChange,
    handleSend,
    handleKeyDown,
    clearChat,
    handleMediaReference,
    removeMediaReference,
  } = useChat();

  return (
    <div className="h-full flex flex-col bg-panel">
      {/* Header */}
      <div className="p-3 pb-2 bg-panel border-b border-border/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4 text-primary" />
            <span className="text-sm font-medium">AI Assistant</span>
          </div>
          {messages.length > 0 && (
            <Button
              size="sm"
              variant="outline"
              onClick={clearChat}
              className="h-7 px-2 text-xs"
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-h-0">
        <ScrollArea className="flex-1 px-3">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-8">
              <MessageCircle className="h-12 w-12 text-muted-foreground/50 mb-4" />
              <h3 className="text-sm font-medium text-foreground mb-2">
                AI Video Assistant
              </h3>
              <p className="text-xs text-muted-foreground max-w-[200px] leading-relaxed">
                Ask me anything about video editing, effects, or get help with your project.
              </p>
            </div>
          ) : (
            <div className="py-3">
              <ChatMessages messages={messages} />
            </div>
          )}
        </ScrollArea>

        {/* Error message */}
        {error && (
          <div className="mx-3 mb-2 bg-destructive/10 border border-destructive/20 text-destructive px-3 py-2 rounded-md text-xs">
            <div className="flex items-start gap-2">
              <svg className="h-4 w-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="flex-1">
                <strong className="font-medium">Error:</strong> {error}
              </div>
            </div>
          </div>
        )}

        {/* Input area */}
        <div className="p-3 pt-2 border-t border-border/50">
          <EnhancedChatInput
            value={inputText}
            onChange={handleInputChange}
            onSend={handleSend}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            referencedMedia={referencedMedia}
            onMediaReference={handleMediaReference}
            onRemoveMediaReference={removeMediaReference}
          />
        </div>
      </div>
    </div>
  );
}
