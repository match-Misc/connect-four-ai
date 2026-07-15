import { useEffect, useState, useRef, type MouseEvent } from 'react'
import { useLocation, useNavigate, Routes, Route, Navigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card'
import { Button } from './components/ui/button'
import { Slider } from './components/ui/slider'
import { Label } from './components/ui/label'
import { Input } from './components/ui/input'
import { Camera, Settings, Monitor, RefreshCw, Save, CheckCircle2 } from 'lucide-react'

function SliderWithInput({ label, value, min = 0, max = 100, step = 1, onChange }: { label: string, value: number, min?: number, max?: number, step?: number, onChange: (v: number) => void }) {
  return (
    <div className="space-y-4 p-5 bg-white rounded-xl border border-slate-200 hover:border-[#b1ca21]/50 hover:bg-slate-50 transition-all duration-300 shadow-sm">
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
          className="flex-1 cursor-pointer [&_[role=slider]]:bg-[#b1ca21] [&_[role=slider]]:border-[#b1ca21] [&_[data-orientation=horizontal]>span:first-child]:bg-slate-200 [&_[data-orientation=horizontal]>span:first-child>span]:bg-[#b1ca21]"
        />
        <Input 
          type="number" 
          value={value} 
          min={min}
          max={max}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-24 bg-white border-slate-200 focus-visible:ring-[#b1ca21] text-center font-mono text-slate-800"
        />
      </div>
    </div>
  )
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  
  const [status, setStatus] = useState<any>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const depthRef = useRef<HTMLImageElement>(null)
  const activeTab = location.pathname.includes('realsense') ? 'realsense' : 'detection'
  const [realsenseSubTab, setRealsenseSubTab] = useState<'sensor' | 'filtering'>('sensor')
  const [measuredDepth, setMeasuredDepth] = useState<number | null>(null)
  const [advancedSettings, setAdvancedSettings] = useState({
    exp_min: 1000, exp_max: 8000, exp_step: 1000,
    gain_min: 16, gain_max: 128, gain_step: 16,
    laser_min: 150, laser_max: 360, laser_step: 50
  })

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status')
      const data = await res.json()
      setStatus(data)
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchStatus()
    const int = setInterval(fetchStatus, 500)
    return () => clearInterval(int)
  }, [])

  const combinationsCount = 
    Math.max(1, Math.floor((advancedSettings.exp_max - advancedSettings.exp_min) / advancedSettings.exp_step) + 1) *
    Math.max(1, Math.floor((advancedSettings.gain_max - advancedSettings.gain_min) / advancedSettings.gain_step) + 1) *
    Math.max(1, Math.floor((advancedSettings.laser_max - advancedSettings.laser_min) / advancedSettings.laser_step) + 1)
    
  const estimatedTime = (combinationsCount * 0.45).toFixed(1)

  const handleImageClick = async (e: MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current) return
    const rect = imageRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const scaleX = imageRef.current.naturalWidth / rect.width
    const scaleY = imageRef.current.naturalHeight / rect.height
    
    await fetch('/api/click', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: Math.round(x * scaleX), y: Math.round(y * scaleY) })
    })
    fetchStatus()
  }

  const handleDepthClick = async (e: MouseEvent<HTMLImageElement>) => {
    if (!depthRef.current) return
    const rect = depthRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const scaleX = depthRef.current.naturalWidth / rect.width
    const scaleY = depthRef.current.naturalHeight / rect.height
    
    const res = await fetch('/api/depth_measure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x: Math.round(x * scaleX), y: Math.round(y * scaleY) })
    })
    const data = await res.json()
    setMeasuredDepth(data.depth_mm)
  }

  const updateDetection = async (key: string, value: number) => {
    await fetch('/api/update_detection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value })
    })
    fetchStatus()
  }

  const cancelAutocalibrate = async () => {
    await fetch('/api/autocalibrate/cancel', { method: 'POST' })
    fetchStatus()
  }

  const updateRealSense = async (key: string, value: number) => {
    await fetch('/api/update_realsense', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value })
    })
    fetchStatus()
  }

  const action = async (endpoint: string, data?: any) => {
    await fetch(`/api/${endpoint}`, { 
      method: 'POST',
      headers: data ? { 'Content-Type': 'application/json' } : undefined,
      body: data ? JSON.stringify(data) : undefined
    })
    fetchStatus()
  }

  if (!status) return <div className="min-h-screen bg-slate-50 flex items-center justify-center text-[#b1ca21] animate-pulse text-lg tracking-widest font-light">INITIALIZING...</div>

  return (
    <div className="flex h-screen bg-slate-100 text-slate-900 overflow-hidden font-sans selection:bg-[#b1ca21]/30">
      
      {/* Sidebar */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col z-10 shadow-lg relative">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-100/50 to-transparent pointer-events-none" />
        <div className="p-8">
          <h1 className="text-2xl font-bold text-slate-800 tracking-tight flex items-center gap-3">
            <Monitor className="w-7 h-7 text-[#b1ca21]" />
            Calibrate
          </h1>
          <p className="text-slate-500 text-xs mt-2 uppercase tracking-widest font-medium">Connect Four AI</p>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4 relative z-10">
          <button 
            onClick={() => navigate('/detection-config')}
            className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-200 group ${activeTab === 'detection' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Camera className={`w-5 h-5 ${activeTab === 'detection' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            Detection Config
          </button>
          
          <button 
            onClick={() => navigate('/realsense-config')}
            className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-200 group ${activeTab === 'realsense' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Settings className={`w-5 h-5 ${activeTab === 'realsense' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            RealSense Calibration
          </button>
        </nav>
        
        <Routes>
          <Route path="/" element={<Navigate to="/detection-config" replace />} />
          <Route path="*" element={null} />
        </Routes>

        <div className="p-6 relative z-10">
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 shadow-sm space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-200">
              <div className="w-2 h-2 rounded-full bg-[#b1ca21] animate-pulse" />
              <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">System Status</span>
            </div>
            
            <ul className="space-y-2 text-xs font-medium">
              <li className="flex items-center justify-between">
                <span className="text-slate-500">Camera Feed</span>
                <span className="text-[#b1ca21] flex items-center"><CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Active</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-slate-500">Corners Defined</span>
                {status.corners?.length === 4 ? (
                  <span className="text-[#b1ca21] flex items-center"><CheckCircle2 className="w-3.5 h-3.5 mr-1" /> 4/4</span>
                ) : (
                  <span className="text-slate-400 flex items-center">{status.corners?.length || 0}/4</span>
                )}
              </li>
              <li className="flex items-center justify-between">
                <span className="text-slate-500">Colors Calibrated</span>
                {status.calibration_complete ? (
                  <span className="text-[#b1ca21] flex items-center"><CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Yes</span>
                ) : (
                  <span className="text-slate-400 flex items-center">No</span>
                )}
              </li>
            </ul>

            <div className="pt-2 border-t border-slate-200 mt-2">
              <p className="text-[10px] text-slate-400 font-mono leading-relaxed break-words" title={status.status_text}>
                &gt; {status.status_text}
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto bg-slate-50/50 relative">
        <div className="p-10 max-w-5xl mx-auto w-full relative z-10">
          
          {activeTab === 'detection' && (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              <header className="mb-6">
                <h2 className="text-3xl font-bold text-slate-800 tracking-tight">Detection Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl">Define the game board corners and fine-tune image processing parameters to ensure reliable token detection.</p>
              </header>

              {/* Camera Feed */}
              <Card className="bg-white border-slate-200 shadow-md overflow-hidden p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img 
                    ref={imageRef}
                    src="/frame/color" 
                    alt="Color Feed" 
                    className="w-full h-auto cursor-crosshair relative z-0"
                    onClick={handleImageClick}
                  />
                  <div className="absolute top-4 left-4 bg-white/90 text-slate-800 text-xs px-3 py-1.5 rounded-full font-medium shadow-md flex items-center gap-2 z-20">
                     <div className="w-1.5 h-1.5 rounded-full bg-[#b1ca21] animate-pulse" />
                     {status.corners?.length < 4 ? (
                      <span>
                        Click to set corner {status.corners.length + 1}/4
                      </span>
                    ) : (
                      <span>
                        Click near a corner to move it
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-3 px-2">
                  <div className="text-xs text-slate-400">
                    Click image to interact with corners.
                  </div>
                  <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg shadow-sm">
                    <Label className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Hole Occupancy Overlay</Label>
                    <button 
                      onClick={() => action('toggle_occupancy')} 
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${status.show_occupancy_overlay ? 'bg-[#b1ca21]' : 'bg-slate-300'}`}
                    >
                      <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${status.show_occupancy_overlay ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                  </div>
                </div>
              </Card>

              {/* Action Buttons */}
              <div className="flex gap-4 p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
                <Button 
                  variant="outline" 
                  onClick={() => action('reset')} 
                  className="flex-1 bg-white border-slate-200 hover:bg-slate-50 text-slate-700 h-12"
                >
                  <RefreshCw className="w-4 h-4 mr-2" /> Reset Corners
                </Button>
                <Button 
                  onClick={() => action('calibrate_colors')} 
                  disabled={status.corners?.length < 4}
                  className="flex-1 bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md shadow-[#b1ca21]/20 h-12 transition-all"
                >
                  <CheckCircle2 className="w-4 h-4 mr-2" /> Calibrate Colors
                </Button>
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm md:col-span-2">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Calibrated Colors</CardTitle>
                    <CardDescription className="text-slate-500">Colors detected for Player 1 and Player 2 tokens.</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1 p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                      <span className="font-medium text-slate-700">Player 1</span>
                      {status.player1_color ? (
                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono text-slate-500">BGR: [{status.player1_color.join(', ')}]</span>
                          <div className="w-8 h-8 rounded-full border-2 border-slate-300 shadow-inner" style={{ backgroundColor: `rgb(${status.player1_color[2]}, ${status.player1_color[1]}, ${status.player1_color[0]})` }} />
                        </div>
                      ) : (
                        <span className="text-xs text-slate-400 italic">Not calibrated</span>
                      )}
                    </div>
                    <div className="flex-1 p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                      <span className="font-medium text-slate-700">Player 2</span>
                      {status.player2_color ? (
                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono text-slate-500">BGR: [{status.player2_color.join(', ')}]</span>
                          <div className="w-8 h-8 rounded-full border-2 border-slate-300 shadow-inner" style={{ backgroundColor: `rgb(${status.player2_color[2]}, ${status.player2_color[1]}, ${status.player2_color[0]})` }} />
                        </div>
                      ) : (
                        <span className="text-xs text-slate-400 italic">Not calibrated</span>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Board Geometry</CardTitle>
                    <CardDescription className="text-slate-500">Adjust the physical mapping.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <SliderWithInput label="Hole Diameter (px)" value={status.hole_diameter} max={150} onChange={(v) => updateDetection('hole_diameter', v)} />
                    <div className="pt-2 border-t border-slate-100">
                      <SliderWithInput label="Occupancy Threshold (Ratio)" value={status.occupancy_threshold || 0.3} min={0} max={1} step={0.05} onChange={(v) => updateDetection('occupancy_threshold', v)} />
                      <p className="text-[11px] text-slate-500 mt-2 px-1 leading-tight mb-4">
                        Minimum percentage of valid pixels needed in the hole to consider it blocked by a token. <br/>
                        <span className="font-semibold">Lower values:</span> More sensitive to tokens, but might trigger on noise. <br/>
                        <span className="font-semibold">Higher values:</span> Requires a more solid reading to detect a token.
                      </p>
                      
                      <SliderWithInput label="Temporal Smoothing (Frames)" value={status.temporal_smoothing || 10} min={1} max={30} step={1} onChange={(v) => updateDetection('temporal_smoothing', v)} />
                      <p className="text-[11px] text-slate-500 mt-2 px-1 leading-tight">
                        Number of frames to consider for stability. Prevents flickering. <br/>
                        <span className="font-semibold">Higher values:</span> Extremely stable, ignores single-frame noise glitches. <br/>
                        <span className="font-semibold">Lower values:</span> Reacts faster to changes, but more prone to flickering.
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Image Filtering</CardTitle>
                    <CardDescription className="text-slate-500">Enhance token colors before detection.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <SliderWithInput label="Contrast" value={status.contrast} max={300} onChange={(v) => updateDetection('contrast', v)} />
                    <SliderWithInput label="Saturation" value={status.saturation} max={300} onChange={(v) => updateDetection('saturation', v)} />
                    <SliderWithInput label="Brightness" value={status.brightness} min={-100} max={100} onChange={(v) => updateDetection('brightness', v)} />
                  </CardContent>
                </Card>
              </div>

              <div className="flex justify-end pt-4 pb-12">
                <Button 
                  size="lg" 
                  onClick={() => action('save_detection')}
                  className="bg-slate-800 hover:bg-slate-700 text-white shadow-md px-8"
                >
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Detection Profile
                </Button>
              </div>
            </div>
          )}

          {activeTab === 'realsense' && (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
              
              <header className="mb-6">
                <h2 className="text-3xl font-bold text-slate-800 tracking-tight">RealSense Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl">Configure the physical camera hardware. These settings are applied directly to the RealSense sensor.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img 
                    ref={depthRef}
                    src="/frame/depth" 
                    alt="Depth Feed" 
                    className="w-full h-auto cursor-crosshair"
                    onClick={handleDepthClick}
                  />
                  {measuredDepth !== null && (
                    <div className="absolute top-4 right-4 bg-white/90 text-slate-800 text-sm px-4 py-2 rounded-xl font-mono shadow-md flex items-center gap-3 z-20 border border-slate-200">
                      <div className="w-2 h-2 rounded-full bg-[#b1ca21] animate-pulse" />
                      Measured Depth: <span className="font-bold text-[#b1ca21]">{measuredDepth} mm</span>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center justify-between mt-3 px-2">
                  <div className="text-xs text-slate-400">
                    Click anywhere on the depth image to measure distance at that point.
                  </div>
                  <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg shadow-sm">
                    <Label className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Hole Occupancy Overlay</Label>
                    <button 
                      onClick={() => action('toggle_occupancy')} 
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${status.show_occupancy_overlay ? 'bg-[#b1ca21]' : 'bg-slate-300'}`}
                    >
                      <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${status.show_occupancy_overlay ? 'translate-x-5' : 'translate-x-1'}`} />
                    </button>
                  </div>
                </div>
              </Card>

              <div className="flex justify-center my-6">
                <div className="inline-flex bg-slate-100 p-1.5 rounded-2xl border border-slate-200 shadow-sm relative">
                  <button 
                    onClick={() => setRealsenseSubTab('sensor')}
                    className={`relative z-10 px-8 py-2.5 text-sm font-semibold rounded-xl transition-all duration-300 ${realsenseSubTab === 'sensor' ? 'text-[#8a9e19] shadow-sm bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'}`}
                  >
                    Sensor Control
                  </button>
                  <button 
                    onClick={() => setRealsenseSubTab('filtering')}
                    className={`relative z-10 px-8 py-2.5 text-sm font-semibold rounded-xl transition-all duration-300 ${realsenseSubTab === 'filtering' ? 'text-[#8a9e19] shadow-sm bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'}`}
                  >
                    Depth Filtering
                  </button>
                </div>
              </div>

              {realsenseSubTab === 'sensor' && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <Card className="bg-white border-slate-200 shadow-sm p-6">
                    <div className="flex flex-col md:flex-row justify-between gap-4">
                      <div className="flex-1">
                        {status.autocalibrate_state === 0 && (
                          <div className="flex flex-col gap-3">
                            <div className="text-sm font-semibold text-slate-700">Auto Calibration Options</div>
                            <div className="flex flex-wrap items-center gap-4">
                              <Button 
                                size="lg" 
                                onClick={() => action('autocalibrate_single', advancedSettings)}
                                disabled={status.corners?.length < 4}
                                className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md shadow-[#b1ca21]/20 px-6 transition-all"
                              >
                                <RefreshCw className="w-4 h-4 mr-2 text-white" /> Quick Calibrate
                              </Button>
                              <Button 
                                size="lg" 
                                onClick={() => action('autocalibrate_step1', advancedSettings)}
                                disabled={status.corners?.length < 4}
                                variant="outline"
                                className="border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21]/10 px-6 transition-all"
                              >
                                Thorough Calibrate (2-Step)
                              </Button>
                            </div>
                            {status.corners?.length < 4 ? (
                              <span className="text-xs text-slate-400">Requires Detection Corners to be set</span>
                            ) : (
                              <span className="text-xs text-slate-400">
                                <strong>Quick</strong> optimizes for current state. <strong>Thorough</strong> requires an empty board for Step 1.<br/>
                                <span className="text-[#8a9e19]">Estimated time per scan: ~{estimatedTime}s</span>
                              </span>
                            )}
                            
                            <div className="mt-4 p-4 border border-slate-200 rounded-lg bg-slate-50/50 space-y-4">
                              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Advanced Sweep Parameters</div>
                              
                              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="space-y-2">
                                  <Label className="text-xs text-slate-500">Exposure Range</Label>
                                  <div className="flex gap-2">
                                    <Input type="number" value={advancedSettings.exp_min} onChange={e => setAdvancedSettings({...advancedSettings, exp_min: Number(e.target.value)})} className="h-8 text-xs" placeholder="Min" />
                                    <Input type="number" value={advancedSettings.exp_max} onChange={e => setAdvancedSettings({...advancedSettings, exp_max: Number(e.target.value)})} className="h-8 text-xs" placeholder="Max" />
                                    <Input type="number" value={advancedSettings.exp_step} onChange={e => setAdvancedSettings({...advancedSettings, exp_step: Number(e.target.value)})} className="h-8 text-xs" placeholder="Step" />
                                  </div>
                                </div>
                                
                                <div className="space-y-2">
                                  <Label className="text-xs text-slate-500">Gain Range</Label>
                                  <div className="flex gap-2">
                                    <Input type="number" value={advancedSettings.gain_min} onChange={e => setAdvancedSettings({...advancedSettings, gain_min: Number(e.target.value)})} className="h-8 text-xs" placeholder="Min" />
                                    <Input type="number" value={advancedSettings.gain_max} onChange={e => setAdvancedSettings({...advancedSettings, gain_max: Number(e.target.value)})} className="h-8 text-xs" placeholder="Max" />
                                    <Input type="number" value={advancedSettings.gain_step} onChange={e => setAdvancedSettings({...advancedSettings, gain_step: Number(e.target.value)})} className="h-8 text-xs" placeholder="Step" />
                                  </div>
                                </div>
                                
                                <div className="space-y-2">
                                  <Label className="text-xs text-slate-500">Laser Range</Label>
                                  <div className="flex gap-2">
                                    <Input type="number" value={advancedSettings.laser_min} onChange={e => setAdvancedSettings({...advancedSettings, laser_min: Number(e.target.value)})} className="h-8 text-xs" placeholder="Min" />
                                    <Input type="number" value={advancedSettings.laser_max} onChange={e => setAdvancedSettings({...advancedSettings, laser_max: Number(e.target.value)})} className="h-8 text-xs" placeholder="Max" />
                                    <Input type="number" value={advancedSettings.laser_step} onChange={e => setAdvancedSettings({...advancedSettings, laser_step: Number(e.target.value)})} className="h-8 text-xs" placeholder="Step" />
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {status.autocalibrate_state === 4 && (
                          <div className="flex flex-col gap-3">
                            <div className="flex items-center gap-2">
                              <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                                <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Quick Scanning Board...
                              </Button>
                              <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200 hover:bg-red-50 hover:text-red-600 px-4">
                                Cancel
                              </Button>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-2.5">
                              <div className="bg-[#b1ca21] h-2.5 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                            </div>
                            <span className="text-xs text-slate-400 text-center">{Math.round((status.autocalibrate_progress || 0) * 100)}% Complete</span>
                          </div>
                        )}

                        {status.autocalibrate_state === 1 && (
                          <div className="flex flex-col gap-3">
                            <div className="flex items-center gap-2">
                              <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                                <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Step 1: Scanning Empty Board...
                              </Button>
                              <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200 hover:bg-red-50 hover:text-red-600 px-4">
                                Cancel
                              </Button>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-2.5">
                              <div className="bg-[#b1ca21] h-2.5 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                            </div>
                            <span className="text-xs text-slate-400 text-center">{Math.round((status.autocalibrate_progress || 0) * 100)}% Complete</span>
                          </div>
                        )}

                        {status.autocalibrate_state === 2 && (
                          <div className="flex flex-col gap-2">
                            <div className="flex items-center gap-4">
                              <Button 
                                size="lg" 
                                onClick={() => action('autocalibrate_step2')}
                                className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md shadow-[#b1ca21]/20 px-6 transition-all animate-pulse"
                              >
                                <RefreshCw className="w-4 h-4 mr-2 text-white" /> Scan Filled Board (Step 2)
                              </Button>
                              <Button variant="outline" onClick={cancelAutocalibrate} className="text-red-500 border-red-200 hover:bg-red-50 hover:text-red-600">
                                Cancel
                              </Button>
                              <span className="text-xs text-slate-600 font-medium max-w-[200px] leading-tight border-l-2 border-[#b1ca21] pl-3">
                                Step 1 Complete! Now, fill the board with several tokens (mix of both colors), then click to scan.
                              </span>
                            </div>
                          </div>
                        )}

                        {status.autocalibrate_state === 3 && (
                          <div className="flex flex-col gap-3">
                            <div className="flex items-center gap-2">
                              <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                                <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Step 2: Scanning Filled Board...
                              </Button>
                              <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200 hover:bg-red-50 hover:text-red-600 px-4">
                                Cancel
                              </Button>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-2.5">
                              <div className="bg-[#b1ca21] h-2.5 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                            </div>
                            <span className="text-xs text-slate-400 text-center">{Math.round((status.autocalibrate_progress || 0) * 100)}% Complete</span>
                          </div>
                        )}
                        
                        {status.autocalibrate_state === 0 && status.autocalibrate_results && status.autocalibrate_results.length > 0 && (
                          <div className="mt-8 border border-slate-200 rounded-xl overflow-hidden shadow-sm">
                            <div className="bg-slate-50 px-4 py-3 border-b border-slate-200 font-semibold text-sm text-slate-700">
                              Top Calibration Results
                            </div>
                            <div className="overflow-x-auto">
                              <table className="w-full text-sm text-left">
                                <thead className="text-xs text-slate-500 bg-white border-b border-slate-100 uppercase">
                                  <tr>
                                    <th className="px-4 py-3">Exposure</th>
                                    <th className="px-4 py-3">Gain</th>
                                    <th className="px-4 py-3">Laser</th>
                                    <th className="px-4 py-3">Score (Cov)</th>
                                    <th className="px-4 py-3">Variance</th>
                                    <th className="px-4 py-3 text-right">Action</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {status.autocalibrate_results.map((res: any, idx: number) => (
                                    <tr key={idx} className={`border-b border-slate-50 hover:bg-slate-50/50 ${idx === 0 ? 'bg-[#b1ca21]/5' : 'bg-white'}`}>
                                      <td className="px-4 py-2.5 font-medium">{res.exposure}</td>
                                      <td className="px-4 py-2.5">{res.gain}</td>
                                      <td className="px-4 py-2.5">{res.laser}</td>
                                      <td className="px-4 py-2.5">
                                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${res.score === 42 ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                          {res.score}/42
                                        </span>
                                      </td>
                                      <td className="px-4 py-2.5 font-mono text-xs text-slate-500">{res.var.toFixed(1)}</td>
                                      <td className="px-4 py-2.5 text-right">
                                        <Button 
                                          size="sm" 
                                          variant="outline"
                                          className={`h-7 text-xs px-3 ${idx === 0 ? 'border-[#b1ca21] text-[#8a9e19] bg-white' : ''}`}
                                          onClick={() => {
                                            updateRealSense('exposure', res.exposure);
                                            updateRealSense('gain', res.gain);
                                            updateRealSense('laser_power', res.laser);
                                          }}
                                        >
                                          Apply
                                        </Button>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Card>

                  <Card className="bg-white border-slate-200 shadow-sm">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                      <CardTitle className="text-lg text-slate-800">Manual Overrides</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <SliderWithInput label="Exposure" value={status.exposure} min={1} max={10000} step={50} onChange={(v) => updateRealSense('exposure', v)} />
                      <SliderWithInput label="Gain" value={status.gain} min={16} max={248} onChange={(v) => updateRealSense('gain', v)} />
                      <SliderWithInput label="Laser Power" value={status.laser_power} min={0} max={360} onChange={(v) => updateRealSense('laser_power', v)} />
                      <SliderWithInput label="Visual Preset" value={status.visual_preset} min={0} max={5} onChange={(v) => updateRealSense('visual_preset', v)} />
                    </CardContent>
                  </Card>
                </div>
              )}

              {realsenseSubTab === 'filtering' && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <Card className="bg-white border-slate-200 shadow-sm flex flex-col">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4 flex flex-row items-center justify-between">
                      <CardTitle className="text-lg text-slate-800">Depth Filtering</CardTitle>
                      
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <SliderWithInput label="Min Depth (mm)" value={status.min_depth} max={5000} step={10} onChange={(v) => updateRealSense('min_depth', v)} />
                      <SliderWithInput label="Max Depth (mm)" value={status.max_depth} max={5000} step={10} onChange={(v) => updateRealSense('max_depth', v)} />
                      
                      <div className="mt-6 p-5 bg-white rounded-xl border border-slate-200 flex items-center justify-between hover:border-[#b1ca21]/50 transition-colors shadow-sm">
                        <Label className="text-slate-700 font-medium tracking-wide text-sm cursor-pointer" htmlFor="emitter-toggle">
                          Emitter Enabled
                        </Label>
                        <input 
                          id="emitter-toggle"
                          type="checkbox" 
                          checked={status.emitter === 1} 
                          onChange={(e) => updateRealSense('emitter', e.target.checked ? 1 : 0)} 
                          className="w-5 h-5 rounded bg-white border-slate-300 text-[#b1ca21] focus:ring-[#b1ca21]/30 cursor-pointer checked:bg-[#b1ca21] checked:border-[#b1ca21]" 
                        />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              <div className="flex justify-end pt-4 pb-12">
                <Button 
                  size="lg" 
                  onClick={() => action('save_realsense')}
                  disabled={status.autocalibrate_state !== 0}
                  className="bg-slate-800 hover:bg-slate-700 text-white shadow-md px-8 shrink-0"
                >
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save RealSense Profile
                </Button>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  )
}
