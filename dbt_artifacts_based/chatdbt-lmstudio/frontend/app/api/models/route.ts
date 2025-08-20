import { NextResponse } from 'next/server'

export async function GET() {
  try {
    const response = await fetch('http://localhost:8000/models')
    const data = await response.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching models:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch models from backend',
        models: [],
        current_model: 'unknown'
      },
      { status: 500 }
    )
  }
}
