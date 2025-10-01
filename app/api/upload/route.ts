import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return NextResponse.json({
        error: 'No file uploaded'
      }, { status: 400 })
    }

    if (!file.name.endsWith('.csv')) {
      return NextResponse.json({
        error: 'Please upload a CSV file'
      }, { status: 400 })
    }

    // Convert file to buffer
    const buffer = Buffer.from(await file.arrayBuffer())
    
    // Run Python script for file analysis and prediction
    const pythonScript = path.join(process.cwd(), 'scripts', 'analyze_upload.py')
    const pythonExecutable = path.join(process.cwd(), 'venv', 'bin', 'python')
    const pythonProcess = spawn(pythonExecutable, [pythonScript], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    // Send file data to Python script
    pythonProcess.stdin.write(buffer)
    pythonProcess.stdin.end()

    let output = ''
    let error = ''

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString()
    })

    return new Promise((resolve) => {
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            // Extract JSON from output (remove any extra text)
            const lines = output.trim().split('\n')
            const jsonLine = lines.find(line => line.startsWith('{'))
            if (jsonLine) {
              const result = JSON.parse(jsonLine)
              resolve(NextResponse.json(result))
            } else {
              resolve(NextResponse.json({
                error: 'No JSON found in output',
                output: output
              }, { status: 500 }))
            }
          } catch (parseError) {
            resolve(NextResponse.json({
              error: 'Failed to parse analysis result',
              output: output
            }, { status: 500 }))
          }
        } else {
          resolve(NextResponse.json({
            error: `Analysis failed with code ${code}`,
            stderr: error
          }, { status: 500 }))
        }
      })
    })

  } catch (error) {
    console.error('Upload API error:', error)
    return NextResponse.json({
      error: 'Internal server error'
    }, { status: 500 })
  }
}
