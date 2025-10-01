import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const { predictionData } = await request.json()
    
    if (!predictionData) {
      return NextResponse.json({ error: 'No prediction data provided' }, { status: 400 })
    }

    // Use virtual environment Python
    const pythonExecutable = path.join(process.cwd(), 'venv', 'bin', 'python')
    const pythonScript = path.join(process.cwd(), 'scripts', 'generate_ai_description.py')
    
    return new Promise((resolve) => {
      const pythonProcess = spawn(pythonExecutable, [pythonScript], {
        stdio: ['pipe', 'pipe', 'pipe']
      })

      let output = ''
      let errorOutput = ''

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString()
      })

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            // Parse JSON output
            const lines = output.trim().split('\n')
            const jsonLine = lines.find(line => line.startsWith('{'))
            
            if (jsonLine) {
              const result = JSON.parse(jsonLine)
              resolve(NextResponse.json(result))
            } else {
              resolve(NextResponse.json({ 
                error: 'Invalid output format',
                output: output,
                errorOutput: errorOutput
              }, { status: 500 }))
            }
          } catch (parseError) {
            resolve(NextResponse.json({ 
              error: 'Failed to parse output',
              output: output,
              errorOutput: errorOutput
            }, { status: 500 }))
          }
        } else {
          resolve(NextResponse.json({ 
            error: 'Python script failed',
            output: output,
            errorOutput: errorOutput
          }, { status: 500 }))
        }
      })

      // Send prediction data to Python script
      pythonProcess.stdin.write(JSON.stringify(predictionData))
      pythonProcess.stdin.end()
    })

  } catch (error) {
    console.error('Error in AI description API:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
