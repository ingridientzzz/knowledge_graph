'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  timestamp: Date
}

interface ChatInterfaceProps {
  onSendMessage?: (message: string) => void
}

export default function ChatInterface({ onSendMessage }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [currentModel, setCurrentModel] = useState<string>('local-model')
  const [isSwitchingModel, setIsSwitchingModel] = useState(false)
  const [contextSize, setContextSize] = useState<number>(20000)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentRequestIdRef = useRef<string | null>(null)

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Check backend connection and load models on component mount
  useEffect(() => {
    const initializeApp = async () => {
      await checkConnection()
      await loadModels()
    }
    initializeApp()
  }, [])

  const loadModels = async () => {
    try {
      console.log('Loading models from /api/models...')
      const response = await fetch('/api/models')
      const data = await response.json()
      console.log('Models response:', data)
      
      if (data.success) {
        setModels(data.models || [])
        setCurrentModel(data.current_model || 'local-model')
        console.log('Loaded models:', data.models)
      } else {
        console.error('Failed to load models:', data.error)
      }
    } catch (error) {
      console.error('Error loading models:', error)
    }
  }

  const checkConnection = async () => {
    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        const data = await response.json()
        setIsConnected(data.status === 'healthy')
      }
    } catch (error) {
      setIsConnected(false)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const messageToSend = input.trim()
    setInput('')
    setIsLoading(true)
    setIsStreaming(false)

    // Generate request ID for backend cancellation
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    currentRequestIdRef.current = requestId
    console.log(`🚀 Starting request: ${requestId}`)

    // Call optional callback
    onSendMessage?.(messageToSend)

    try {
      // Create AbortController for cancellation
      const abortController = new AbortController()
      abortControllerRef.current = abortController
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': requestId, // Send request ID to backend
        },
        body: JSON.stringify({ message: messageToSend, context_size: contextSize }),
        signal: abortController.signal,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      
      if (!reader) {
        throw new Error('Failed to get response reader')
      }

      // Create initial assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        sources: [],
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Clear loading state and start streaming
      setIsLoading(false)
      setIsStreaming(true)

      let buffer = ''
      let fullResponse = ''
      let lastMessageIndex = -1

      // Get the index of the message we just added
      setMessages(prev => {
        lastMessageIndex = prev.length - 1
        return prev
      })

      console.log('Starting to read stream...')
      
      try {
        while (true) {
          const { done, value } = await reader.read()
          
          if (done) {
            console.log('Stream reading completed')
            break
          }

          // Decode the chunk
          const chunk = decoder.decode(value, { stream: true })
          console.log('Received chunk:', chunk)
          
          buffer += chunk
          
          // Process complete lines
          const lines = buffer.split('\n')
          buffer = lines.pop() || '' // Keep last incomplete line

          for (const line of lines) {
            if (line.trim() === '') continue
            
            if (line.startsWith('data: ')) {
              try {
                const jsonStr = line.slice(6).trim()
                if (!jsonStr) continue
                
                const data = JSON.parse(jsonStr)
                console.log('Parsed SSE data:', data)
                
                if (data.token && !data.done) {
                  // Real-time streaming token
                  fullResponse += data.token
                  console.log('Adding token:', data.token, 'Full response so far:', fullResponse.substring(0, 50) + '...')
                  
                  // Force immediate UI update
                  setMessages(prev => {
                    const newMessages = [...prev]
                    if (lastMessageIndex >= 0 && lastMessageIndex < newMessages.length) {
                      newMessages[lastMessageIndex] = {
                        ...newMessages[lastMessageIndex],
                        content: fullResponse
                      }
                    }
                    return newMessages
                  })
                  
                  // Force immediate scroll
                  setTimeout(() => scrollToBottom(), 0)
                  
                } else if (data.done) {
                  // Completion
                  console.log('Stream completed with sources:', data.sources)
                  if (data.response) {
                    fullResponse = data.response
                  }
                  
                  setMessages(prev => {
                    const newMessages = [...prev]
                    if (lastMessageIndex >= 0 && lastMessageIndex < newMessages.length) {
                      newMessages[lastMessageIndex] = {
                        ...newMessages[lastMessageIndex],
                        content: fullResponse,
                        sources: data.sources || []
                      }
                    }
                    return newMessages
                  })
                  
                  setIsStreaming(false)
                  abortControllerRef.current = null
                  return // Exit the function
                  
                } else if (data.error) {
                  console.error('Received error:', data.error)
                  setIsStreaming(false)
                  abortControllerRef.current = null
                  throw new Error(data.error)
                }
                
              } catch (parseError) {
                console.warn('Failed to parse SSE data:', line, parseError)
              }
            }
          }
        }
      } catch (streamError) {
        console.error('Stream reading error:', streamError)
        setIsStreaming(false)
        throw streamError
      }

    } catch (error) {
      console.error('Chat error:', error)
      
      // Handle abortion gracefully
      if (error instanceof Error && error.name === 'AbortError') {
        const abortMessage: Message = {
          role: 'assistant',
          content: '🛑 Request cancelled by user.',
          timestamp: new Date()
        }
        setMessages(prev => [...prev, abortMessage])
      } else {
        const errorMessage: Message = {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please make sure LM Studio is running and the backend server is started.`,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, errorMessage])
      }
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      abortControllerRef.current = null
      currentRequestIdRef.current = null
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  const stopGeneration = async () => {
    console.log('🛑 Stop button clicked')
    
    // 1. Stop frontend immediately
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    
    // 2. Try to cancel backend process
    if (currentRequestIdRef.current) {
      try {
        console.log(`🛑 Cancelling backend request: ${currentRequestIdRef.current}`)
        const cancelResponse = await fetch(`/api/cancel/${currentRequestIdRef.current}`, {
          method: 'POST',
        })
        const cancelResult = await cancelResponse.json()
        console.log('🛑 Cancel response:', cancelResult)
      } catch (error) {
        console.error('❌ Failed to cancel backend request:', error)
      }
      currentRequestIdRef.current = null
    }
    
    setIsLoading(false)
    setIsStreaming(false)
    
    // Add cancellation message to chat
    setMessages(prev => {
      const newMessages = [...prev]
      if (newMessages.length > 0 && newMessages[newMessages.length - 1].role === 'assistant') {
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          content: newMessages[newMessages.length - 1].content + '\n\n_Request cancelled by user._'
        }
      }
      return newMessages
    })
  }

  const refreshIndex = async () => {
    try {
      setIsLoading(true)
      const response = await fetch('/api/refresh', {
        method: 'POST',
      })

      if (response.ok) {
        const data = await response.json()
        const successMessage: Message = {
          role: 'assistant',
          content: `✅ ${data.message}`,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, successMessage])
      } else {
        throw new Error('Failed to refresh index')
      }
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Error refreshing index: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
    }
  }

  const switchModel = async (modelId: string) => {
    if (modelId === currentModel || isSwitchingModel) return
    
    try {
      setIsSwitchingModel(true)
      console.log(`Switching to model: ${modelId}`)
      
      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`, {
        method: 'POST',
      })
      
      const data = await response.json()
      
      if (data.success) {
        setCurrentModel(modelId)
        const successMessage: Message = {
          role: 'assistant',
          content: `🤖 Successfully switched to model: **${modelId}**\n\nThe chat engine has been reinitialized. You can now ask questions and they'll be processed by the new model!`,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, successMessage])
      } else {
        throw new Error(data.error || 'Failed to switch model')
      }
    } catch (error) {
      console.error('Error switching model:', error)
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Failed to switch to model ${modelId}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsSwitchingModel(false)
    }
  }

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getSourceDisplayName = (source: string) => {
    // Extract model name from source format like "model: model_name (path/to/file)"
    const match = source.match(/^(\w+):\s*([^(]+)/)
    if (match) {
      const [, type, name] = match
      return `${type}: ${name.trim()}`
    }
    return source
  }

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <h2 className="text-lg font-semibold text-gray-800">ChatDBT Knowledge Graph</h2>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-500">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          {isConnected && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">•</span>
              <select 
                value={currentModel}
                onChange={(e) => switchModel(e.target.value)}
                disabled={isSwitchingModel || isLoading || isStreaming}
                className="text-sm bg-gray-100 border border-gray-300 rounded px-2 py-1 text-gray-700 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed max-w-48"
                title="Select AI Model"
              >
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    🤖 {model.id.length > 25 ? `${model.id.substring(0, 25)}...` : model.id}
                  </option>
                ))}
              </select>
              <span className="text-xs text-gray-400">•</span>
              <div className="flex items-center space-x-1">
                <label htmlFor="context-size" className="text-xs text-gray-500">Context:</label>
                <select 
                  id="context-size"
                  value={contextSize}
                  onChange={(e) => setContextSize(Number(e.target.value))}
                  disabled={isLoading || isStreaming}
                  className="text-xs bg-gray-100 border border-gray-300 rounded px-1 py-1 text-gray-700 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Context Size (characters)"
                >
                  <option value={8000}>8K</option>
                  <option value={16000}>16K</option>
                  <option value={20000}>20K</option>
                  <option value={32000}>32K</option>
                  <option value={50000}>50K</option>
                </select>
              </div>
              {isSwitchingModel && (
                <div className="text-xs text-blue-600">Switching...</div>
              )}
            </div>
          )}
        </div>
        
        <div className="flex space-x-2">
          {(isLoading || isStreaming) && (
            <button
              onClick={stopGeneration}
              className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 flex items-center space-x-1"
              title="Stop generation"
            >
              <span>🛑</span>
              <span>Stop</span>
            </button>
          )}
          <button
            onClick={refreshIndex}
            disabled={isLoading || isStreaming || isSwitchingModel}
            className="px-3 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200 disabled:opacity-50"
            title="Refresh knowledge graph index"
          >
            🔄 Refresh
          </button>
          <button
            onClick={clearChat}
            disabled={isLoading || isStreaming || isSwitchingModel}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50"
            title="Clear chat history"
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-20">
            <div className="mb-4">
              <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <p className="text-lg mb-2">Welcome to ChatDBT Knowledge Graph!</p>
            <p className="text-sm mb-4">Ask me anything about your dbt models, dependencies, and data transformations.</p>
            <div className="text-left max-w-lg mx-auto space-y-2 text-sm">
              <p className="font-medium">Try asking:</p>
              <ul className="space-y-1 text-gray-600">
                <li>• "What packages have the most models and tests?"</li>
                <li>• "Show me all models in the bmt package"</li>
                <li>• "What seed tables are available in customer_segments?"</li>
                <li>• "How does accelerator_consulting_survey data flow to other models?"</li>
              </ul>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl ${message.role === 'user' ? 'ml-12' : 'mr-12'}`}>
                <div className={`p-3 rounded-lg ${
                  message.role === 'user' 
                    ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white' 
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  
                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-gray-300">
                      <p className="text-xs font-semibold mb-1">📊 Knowledge Sources:</p>
                      <div className="flex flex-wrap gap-1">
                        {message.sources.slice(0, 5).map((source, idx) => (
                          <span
                            key={idx}
                            className="inline-block px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded"
                            title={source}
                          >
                            {getSourceDisplayName(source)}
                          </span>
                        ))}
                        {message.sources.length > 5 && (
                          <span className="text-xs text-gray-500">
                            +{message.sources.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Timestamp */}
                <div className={`text-xs text-gray-400 mt-1 ${
                  message.role === 'user' ? 'text-right' : 'text-left'
                }`}>
                  {formatTimestamp(message.timestamp)}
                </div>
              </div>
            </div>
          ))
        )}
        
        {/* Loading indicator - only show when initially loading, not during streaming */}
        {isLoading && !isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-3xl mr-12">
              <div className="bg-gray-100 text-gray-800 p-3 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm">Connecting to LM Studio...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-3xl mr-12">
              <div className="bg-blue-50 text-blue-800 p-3 rounded-lg border border-blue-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                    <span className="text-sm">🤖 Generating response...</span>
                  </div>
                  <button
                    onClick={stopGeneration}
                    className="ml-3 px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 flex items-center space-x-1"
                    title="Stop generation"
                  >
                    <span>🛑</span>
                    <span>Stop</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isConnected ? "Ask about your dbt knowledge graph..." : "Backend disconnected. Please check the server."}
              className="w-full p-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-50"
              rows={Math.min(4, Math.max(1, input.split('\n').length))}
              disabled={isLoading || isStreaming || !isConnected}
              style={{ minHeight: '48px', maxHeight: '120px' }}
            />
            
            {/* Character count */}
            {input.length > 500 && (
              <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                {input.length}/2000
              </div>
            )}
          </div>
          
          <button
            onClick={sendMessage}
            disabled={isLoading || isStreaming || !input.trim() || !isConnected}
            className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg hover:from-indigo-600 hover:to-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            {isLoading || isStreaming ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>{isLoading ? 'Sending' : 'Streaming'}</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                <span>Send</span>
              </>
            )}
          </button>
        </div>
        
        {/* Connection status warning */}
        {!isConnected && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
            ⚠️ Cannot connect to backend. Make sure LM Studio is running and the backend server is started.
          </div>
        )}
      </div>
    </div>
  )
}
