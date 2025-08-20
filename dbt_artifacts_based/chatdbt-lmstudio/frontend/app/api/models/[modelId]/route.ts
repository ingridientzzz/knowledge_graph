import { NextRequest, NextResponse } from 'next/server'

export async function POST(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  try {
    const { modelId } = params
    console.log(`Switching to model: ${modelId}`)
    
    const response = await fetch(`http://localhost:8000/models/${encodeURIComponent(modelId)}`, {
      method: 'POST',
    })
    
    const data = await response.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error switching model:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to switch model',
        current_model: 'unknown'
      },
      { status: 500 }
    )
  }
}
