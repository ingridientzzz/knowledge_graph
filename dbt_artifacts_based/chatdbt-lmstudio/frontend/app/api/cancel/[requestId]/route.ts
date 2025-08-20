import { NextRequest, NextResponse } from 'next/server'

export async function POST(
  request: NextRequest,
  { params }: { params: { requestId: string } }
) {
  try {
    const { requestId } = params
    console.log(`🛑 Frontend: Cancelling request ${requestId}`)
    
    // Forward cancellation request to backend
    const backendResponse = await fetch(`http://localhost:8000/cancel/${requestId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    
    const result = await backendResponse.json()
    console.log('🛑 Backend cancel response:', result)
    
    return NextResponse.json(result)
  } catch (error) {
    console.error('❌ Error cancelling request:', error)
    return NextResponse.json(
      { error: 'Failed to cancel request' },
      { status: 500 }
    )
  }
}
