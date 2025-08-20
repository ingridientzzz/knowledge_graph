import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

export async function POST() {
  try {
    // Forward refresh request to backend
    const backendResponse = await fetch(`${BACKEND_URL}/refresh-index`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({}))
      return NextResponse.json(
        { 
          error: errorData.detail || `Backend error: ${backendResponse.status}`,
          status: backendResponse.status
        },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    
    return NextResponse.json({
      message: data.message,
      timestamp: new Date().toISOString()
    })

  } catch (error) {
    console.error('Refresh API Error:', error)
    
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { 
          error: 'Cannot connect to backend server',
          details: 'Check that the backend server is running at ' + BACKEND_URL
        },
        { status: 503 }
      )
    }
    
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}
