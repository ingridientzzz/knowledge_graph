'use client'

import ChatInterface from '../components/ChatInterface'

export default function Home() {
  const handleSendMessage = (message: string) => {
    console.log('Message sent:', message)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      <div className="container mx-auto p-4 h-screen flex flex-col">
        <header className="text-center mb-6">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            ChatDBT with Local LLM
          </h1>
          <p className="text-gray-600">
            Chat with your dbt knowledge graph using local AI models
          </p>
        </header>
        
        <div className="flex-1 max-w-6xl mx-auto w-full">
          <ChatInterface onSendMessage={handleSendMessage} />
        </div>
        
        <footer className="text-center mt-4 text-sm text-gray-500">
          Powered by LM Studio
        </footer>
      </div>
    </div>
  )
}
