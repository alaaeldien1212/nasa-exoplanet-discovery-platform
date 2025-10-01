import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const body = await request.json()
    
    // Validate input data
    const requiredFields = ['koi_period', 'koi_prad', 'koi_steff']
    const missingFields = requiredFields.filter(field => !(field in body))
    
    if (missingFields.length > 0) {
      return NextResponse.json({
        error: `Missing required fields: ${missingFields.join(', ')}`
      }, { status: 400 })
    }
    
    // Run Python prediction script
    const pythonScript = path.join(process.cwd(), 'scripts', 'predict.py')
    const pythonExecutable = path.join(process.cwd(), 'venv', 'bin', 'python')
    const pythonProcess = spawn(pythonExecutable, [pythonScript], {
      stdio: ['pipe', 'pipe', 'pipe']
    })
    
    // Send input data to Python script
    pythonProcess.stdin.write(JSON.stringify(body))
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
              error: 'Failed to parse prediction result',
              output: output
            }, { status: 500 }))
          }
        } else {
          resolve(NextResponse.json({
            error: `Python script failed with code ${code}`,
            stderr: error
          }, { status: 500 }))
        }
      })
    })
    
  } catch (error) {
    console.error('Prediction API error:', error)
    return NextResponse.json({
      error: 'Internal server error'
    }, { status: 500 })
  }
}

export async function GET(): Promise<NextResponse> {
  try {
    // Get model information
    const pythonScript = path.join(process.cwd(), 'scripts', 'model_info.py')
    const pythonExecutable = path.join(process.cwd(), 'venv', 'bin', 'python')
    const pythonProcess = spawn(pythonExecutable, [pythonScript], {
      stdio: ['pipe', 'pipe', 'pipe']
    })
    
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
              error: 'Failed to parse model info',
              output: output
            }, { status: 500 }))
          }
        } else {
          resolve(NextResponse.json({
            error: `Failed to get model info: ${error}`,
            code: code
          }, { status: 500 }))
        }
      })
    })
    
  } catch (error) {
    console.error('Model info API error:', error)
    return NextResponse.json({
      error: 'Internal server error'
    }, { status: 500 })
  }
}
