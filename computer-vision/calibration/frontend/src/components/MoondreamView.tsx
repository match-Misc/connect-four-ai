import { useEffect, useState, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card'
import { Button } from './ui/button'
import { Slider } from './ui/slider'
import { Label } from './ui/label'
import { Input } from './ui/input'
import { Settings, ScanFace, Play, Square, Activity, Loader2 } from 'lucide-react'

function SliderWithInput({ label, value, min = 0, max = 100, step = 1, onChange }: { label: string, value: number, min?: number, max?: number, step?: number, onChange: (v: number) => void }) {
  return (
    <div className="space-y-4 p-5 bg-white rounded-xl border border-slate-200 shadow-sm">
      <Label className="text-slate-700 font-medium tracking-wide text-sm flex items-center gap-2">
        {label}
      </Label>
      <div className="flex items-center gap-6">
        <Slider 
          value={[value]} 
          min={min} 
          max={max} 
          step={step} 
          onValueChange={(v) => onChange(v[0])} 
          className="flex-1 cursor-pointer"
        />
        <Input 
          type="number" 
          value={value} 
          min={min}
          max={max}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-24 bg-white border-slate-200 text-center font-mono text-slate-800"
        />
      </div>
    </div>
  )
}

export default function MoondreamView() {
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  const [isContinuous, setIsContinuous] = useState(false)
  const [intervalMs, setIntervalMs] = useState(3000)
  const [targets, setTargets] = useState("black chip")
  const [apiMode, setApiMode] = useState<"cloud" | "local">("cloud")
  const [apiKey, setApiKey] = useState("")
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  
  const [latency, setLatency] = useState<{ total: number, prep: number, calls: any } | null>(null)
  const [detectionResults, setDetectionResults] = useState<any>(null)

  const captureAndDetect = useCallback(async () => {
    if (!imgRef.current || !canvasRef.current) return

    const img = imgRef.current
    const canvas = canvasRef.current
    
    // Set canvas dimensions to match image natural size
    canvas.width = img.naturalWidth || 640
    canvas.height = img.naturalHeight || 480
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Draw current frame to canvas for Base64 extraction
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
    const base64Image = canvas.toDataURL('image/jpeg', 0.8)
    
    try {
      setIsProcessing(true)
      setErrorMsg(null)
      const res = await fetch('/api/moondream/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: base64Image,
          targets: targets.split(',').map(t => t.trim()).filter(Boolean),
          mode: apiMode,
          api_key: apiKey
        })
      })
      
      const data = await res.json()
      
      if (data.error) {
        console.error("Detection error:", data.error)
        setErrorMsg(data.error)
        return
      }
      
      setLatency(data.latency)
      setDetectionResults(data.detections)
      
      // Clear the canvas and redraw boxes
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      if (data.detections) {
        Object.keys(data.detections).forEach(target => {
          const boxes = data.detections[target]
          
          ctx.strokeStyle = target.includes("green") ? "#b1ca21" : "#FF00FF"
          ctx.fillStyle = target.includes("green") ? "rgba(177, 202, 33, 0.2)" : "rgba(255, 0, 255, 0.2)"
          ctx.lineWidth = 3
          ctx.font = "16px Inter, sans-serif"
          
          boxes.forEach((box: number[]) => {
            const [xmin, ymin, xmax, ymax] = box
            // Moondream returns normalized coords [0, 1]
            const x = xmin * canvas.width
            const y = ymin * canvas.height
            const w = (xmax - xmin) * canvas.width
            const h = (ymax - ymin) * canvas.height
            
            ctx.beginPath()
            ctx.rect(x, y, w, h)
            ctx.fill()
            ctx.stroke()
            
            // Draw label
            ctx.fillStyle = ctx.strokeStyle
            ctx.fillText(target, x, y - 5)
          })
        })
      }
    } catch (err) {
      console.error("Failed to detect:", err)
      setErrorMsg("Network error or timeout while connecting to local model.")
    } finally {
      setIsProcessing(false)
    }
  }, [targets, apiMode, apiKey])

  // Handle continuous polling
  useEffect(() => {
    let timer: number
    if (isContinuous) {
      timer = setInterval(() => {
        captureAndDetect()
      }, intervalMs)
    }
    return () => clearInterval(timer)
  }, [isContinuous, intervalMs, captureAndDetect])

  const getFeasibilityBadge = () => {
    if (!latency) return null
    const total = latency.total
    if (total < 800) return <div className="bg-emerald-100 text-emerald-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border border-emerald-200">Feasible for Real-Time</div>
    if (total <= 2000) return <div className="bg-orange-100 text-orange-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border border-orange-200">Moderate Latency</div>
    return <div className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border border-red-200">Slow (Async Only)</div>
  }

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-12 w-full">
      <header className="mb-6">
        <h2 className="text-3xl font-bold text-slate-800 tracking-tight flex items-center gap-3">
          <ScanFace className="w-8 h-8 text-[#b1ca21]" />
          Moondream AI Vision
        </h2>
        <p className="text-slate-500 mt-2 text-sm max-w-2xl">
          Real-time object detection powered by Moondream running locally.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Camera Feed */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="bg-white border-slate-200 shadow-md p-2">
            <div className="relative group rounded-lg overflow-hidden border border-slate-100 bg-slate-900 aspect-video flex items-center justify-center">
              <>
                <img 
                  ref={imgRef} 
                  src="/frame/raw"
                  alt="RealSense Color Feed"
                  crossOrigin="anonymous"
                  className="absolute inset-0 w-full h-full object-contain"
                />
                <canvas 
                  ref={canvasRef} 
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none z-10"
                />
              </>
            </div>
            
            <div className="flex items-center gap-4 mt-4 px-2">
              <Button 
                onClick={captureAndDetect} 
                disabled={isContinuous || isProcessing}
                className="flex-1 bg-slate-800 hover:bg-slate-700 text-white shadow-md h-12"
              >
                {isProcessing ? (
                  <><Loader2 className="w-4 h-4 mr-2 animate-spin text-[#b1ca21]" /> Processing (60s+)...</>
                ) : (
                  <><ScanFace className="w-4 h-4 mr-2 text-[#b1ca21]" /> Single Shot</>
                )}
              </Button>
              
              <Button 
                onClick={() => setIsContinuous(!isContinuous)}
                className={`flex-1 h-12 shadow-md transition-all ${isContinuous ? 'bg-red-500 hover:bg-red-600 text-white' : 'bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-[#b1ca21]/20'}`}
              >
                {isContinuous ? (
                  <><Square className="w-4 h-4 mr-2" /> Stop Stream</>
                ) : (
                  <><Play className="w-4 h-4 mr-2" /> Auto-Stream</>
                )}
              </Button>
            </div>

            {errorMsg && (
              <div className="mt-4 mx-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 font-mono">
                <span className="font-bold">Error:</span> {errorMsg}
              </div>
            )}
          </Card>
          
          <Card className="bg-white border-slate-200 shadow-sm">
            <CardHeader className="pb-4 border-b border-slate-100 mb-4">
              <CardTitle className="text-lg text-slate-800 flex items-center gap-2">
                <Activity className="w-5 h-5 text-slate-500" />
                Performance Benchmarks
              </CardTitle>
            </CardHeader>
            <CardContent>
              {latency ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                    <span className="text-slate-600 font-medium">Round-Trip Total</span>
                    <div className="flex items-center gap-4">
                      <span className="font-mono text-lg font-bold text-slate-800">{latency.total.toFixed(0)} ms</span>
                      {getFeasibilityBadge()}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                      <div className="text-slate-500 mb-1">Image Prep (Client + Server)</div>
                      <div className="font-mono font-medium text-slate-700">{latency.prep.toFixed(0)} ms</div>
                    </div>
                    {Object.keys(latency.calls).map(target => (
                      <div key={target} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <div className="text-slate-500 mb-1">Model: {target}</div>
                        <div className="font-mono font-medium text-slate-700">{latency.calls[target].toFixed(0)} ms</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-slate-400 italic py-4 text-center">
                  Run a detection to see latency benchmarks.
                </div>
              )}
            </CardContent>
          </Card>

          {/* Detection Results Table */}
          {latency && (
            <Card className="bg-white border-slate-200 shadow-sm mt-6">
              <CardHeader className="pb-4 border-b border-slate-100">
                <CardTitle className="text-lg text-slate-800 flex items-center gap-2">
                  <ScanFace className="w-5 h-5 text-slate-500" />
                  Detection Results
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs text-slate-500 uppercase bg-slate-50">
                      <tr>
                        <th className="px-4 py-3 rounded-tl-lg">Target</th>
                        <th className="px-4 py-3">Found?</th>
                        <th className="px-4 py-3 rounded-tr-lg">Coordinates [x, y, w, h]</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(latency.calls).map(target => {
                        const hasDetections = detectionResults && detectionResults[target] && detectionResults[target].length > 0;
                        return (
                          <tr key={target} className="border-b border-slate-100 last:border-0">
                            <td className="px-4 py-3 font-medium text-slate-800">{target}</td>
                            <td className="px-4 py-3">
                              {hasDetections ? (
                                <span className="text-green-600 font-semibold px-2 py-1 bg-green-50 rounded-md">Yes ({detectionResults[target].length})</span>
                              ) : (
                                <span className="text-slate-400 font-medium">No</span>
                              )}
                            </td>
                            <td className="px-4 py-3 font-mono text-xs text-slate-500 max-w-[200px] overflow-x-auto">
                              {hasDetections ? 
                                detectionResults[target].map((box: number[], i: number) => {
                                  // format is usually [x, y, x2, y2], let's just display it
                                  const isNorm = box[2] <= 1 && box[3] <= 1;
                                  return <div key={i} className="mb-1 last:mb-0">[{box.map(b => isNorm ? b.toFixed(3) : b.toFixed(0)).join(', ')}]</div>
                                })
                              : "-"}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column: Settings */}
        <div className="space-y-6">
          <Card className="bg-white border-slate-200 shadow-sm">
            <CardHeader className="pb-4 border-b border-slate-100 mb-4">
              <CardTitle className="text-lg text-slate-800 flex items-center gap-2">
                <Settings className="w-5 h-5 text-slate-500" />
                Detection Settings
              </CardTitle>
              <CardDescription className="text-slate-500">Configure the Moondream queries and intervals.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="text-slate-700 font-medium tracking-wide text-sm">API Mode</Label>
                <div className="flex bg-slate-100 p-1 rounded-lg">
                  <button 
                    onClick={() => setApiMode("cloud")}
                    className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${apiMode === "cloud" ? "bg-white shadow-sm text-slate-800" : "text-slate-500 hover:text-slate-700"}`}
                  >
                    Cloud API
                  </button>
                  <button 
                    onClick={() => setApiMode("local")}
                    className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${apiMode === "local" ? "bg-white shadow-sm text-slate-800" : "text-slate-500 hover:text-slate-700"}`}
                  >
                    Local (Photon)
                  </button>
                </div>
              </div>

              {apiMode === "cloud" && (
                <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-300">
                  <Label className="text-slate-700 font-medium tracking-wide text-sm flex justify-between">
                    <span>Moondream API Key</span>
                    <a href="https://console.moondream.ai" target="_blank" rel="noreferrer" className="text-xs text-[#b1ca21] hover:underline">Get Key</a>
                  </Label>
                  <Input 
                    type="password"
                    value={apiKey} 
                    onChange={(e) => setApiKey(e.target.value)}
                    className="bg-slate-50 border-slate-200"
                    placeholder="md_..."
                  />
                </div>
              )}

              <div className="space-y-2 pt-2 border-t border-slate-100">
                <Label className="text-slate-700 font-medium tracking-wide text-sm">Target Objects (Comma Separated)</Label>
                <Input 
                  value={targets} 
                  onChange={(e) => setTargets(e.target.value)}
                  className="bg-slate-50"
                  placeholder="e.g. black chip, green chip"
                />
              </div>
              
              <div className="pt-2 border-t border-slate-100">
                <SliderWithInput 
                  label="Auto-Stream Interval (ms)" 
                  value={intervalMs} 
                  min={500} 
                  max={10000} 
                  step={500} 
                  onChange={setIntervalMs} 
                />
                <p className="text-[11px] text-slate-500 mt-2 px-1 leading-tight mb-4">
                  How often to capture a frame and send to the Moondream server during Auto-Stream.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
