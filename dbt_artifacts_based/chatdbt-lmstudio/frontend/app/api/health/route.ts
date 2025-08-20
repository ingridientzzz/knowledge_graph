import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function GET() {
  try {
    // Check backend health
    const backendResponse = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      return NextResponse.json(
        { 
          status: 'unhealthy',
          frontend: 'healthy',
          backend: 'unhealthy',
          backend_status: backendResponse.status,
          error: `Backend returned ${backendResponse.status}`
        },
        { status: 503 }
      )
    }

    const backendHealth = await backendResponse.json()
    
    return NextResponse.json({
      status: 'healthy',
      frontend: 'healthy',
      backend: backendHealth.status || 'healthy',
      backend_details: {
        lm_studio_model: backendHealth.lm_studio_model,
        lm_studio_host: backendHealth.lm_studio_host,
        data_path: backendHealth.data_path,
        index_loaded: backendHealth.index_loaded,
        chat_engine_ready: backendHealth.chat_engine_ready
      },
      backend_url: BACKEND_URL,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('Health check error:', error)
    
    return NextResponse.json(
      { 
        status: 'unhealthy',
        frontend: 'healthy',
        backend: 'unreachable',
        error: error instanceof Error ? error.message : 'Unknown error',
        backend_url: BACKEND_URL,
        timestamp: new Date().toISOString()
      },
      { status: 503 }
    )
  }
}
