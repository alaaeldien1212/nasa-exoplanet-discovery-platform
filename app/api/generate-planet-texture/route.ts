import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
  try {
    const planetData = await request.json()
    
    // Validate required fields
    if (!planetData.radius || !planetData.temperature || !planetData.classification) {
      return NextResponse.json(
        { success: false, error: 'Missing required planet data fields' },
        { status: 400 }
      )
    }

    // Get the Python executable path
    const pythonExecutable = path.join(process.cwd(), 'venv', 'bin', 'python')
    const pythonScript = path.join(process.cwd(), 'scripts', 'ai_planet_generator.py')

    // Spawn Python process
    const pythonProcess = spawn(pythonExecutable, [pythonScript], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    // Send planet data to Python script
    pythonProcess.stdin.write(JSON.stringify(planetData))
    pythonProcess.stdin.end()

    let output = ''
    let errorOutput = ''

    // Collect output
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString()
    })

    // Wait for process to complete
    await new Promise((resolve, reject) => {
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve(output)
        } else {
          reject(new Error(`Python process exited with code ${code}: ${errorOutput}`))
        }
      })
    })

    // Parse the JSON output
    const result = JSON.parse(output.trim())
    
    if (result.success) {
      return NextResponse.json({
        success: true,
        texture: result.image,
        planet_type: result.planet_type
      })
    } else {
      return NextResponse.json(
        { success: false, error: result.error },
        { status: 500 }
      )
    }

  } catch (error) {
    console.error('Error generating planet texture:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to generate planet texture' },
      { status: 500 }
    )
  }
}
