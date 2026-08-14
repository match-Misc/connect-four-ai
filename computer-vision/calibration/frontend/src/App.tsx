import { useEffect, useState, useRef, type MouseEvent } from 'react'
import { useLocation, useNavigate, Routes, Route, Navigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card'
import { Button } from './components/ui/button'
import { Slider } from './components/ui/slider'
import { Label } from './components/ui/label'
import { Input } from './components/ui/input'
import { Settings, Monitor, RefreshCw, Save, CheckCircle2, LayoutGrid, Palette, Target, Nfc } from 'lucide-react'

function SliderWithInput({ label, description, value, min = 0, max = 100, step = 1, onChange }: { label: string, description?: React.ReactNode, value: number, min?: number, max?: number, step?: number, onChange: (v: number) => void }) {
  return (
    <div className="space-y-4 p-5 bg-white rounded-xl border border-slate-200 hover:border-[#b1ca21]/50 hover:bg-slate-50 transition-all duration-300 shadow-sm">
      <div className="space-y-1">
        <Label className="text-slate-700 font-medium tracking-wide text-sm flex items-center gap-2">
          {label}
        </Label>
        {description && <p className="text-xs text-slate-500 leading-tight">{description}</p>}
      </div>
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
  
  const activeTab = 
    location.pathname.includes('realsense') ? 'realsense' : 
    location.pathname.includes('color-calibration') ? 'color-calibration' :
    location.pathname.includes('detection-calibration') ? 'detection-calibration' :
    location.pathname.includes('nfc-testing') ? 'nfc-testing' : 'define-board'

  useEffect(() => {
    let mode = 'define_board'
    if (activeTab === 'realsense') mode = 'realsense'
    if (activeTab === 'color-calibration') mode = 'color_calibration'
    if (activeTab === 'detection-calibration') mode = 'detection_calibration'
    if (activeTab === 'nfc-testing') mode = 'nfc_testing'
    
    fetch('/api/set_ui_mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ui_mode: mode })
    })
  }, [activeTab])
  const [sessionTime] = useState(() => Date.now())
    
  const [realsenseSubTab, setRealsenseSubTab] = useState<'sensor' | 'filtering'>('sensor')
  const [measuredDepth, setMeasuredDepth] = useState<number | null>(null)
  const [advancedSettings, setAdvancedSettings] = useState({
    exp_min: 1000, exp_max: 8000, exp_step: 1500,
    gain_min: 16, gain_max: 128, gain_step: 24,
    laser_min: 150, laser_max: 360, laser_step: 75,
    duration: 3.0
  })
  const [pendingOverrides, setPendingOverrides] = useState<any>({})

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

  const handleImageClick = async (e: MouseEvent<HTMLImageElement>) => {
    if (activeTab !== 'define-board') return
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
    <div className="flex flex-col lg:flex-row h-screen bg-slate-100 text-slate-900 overflow-hidden font-sans selection:bg-[#b1ca21]/30">
      
      {/* Sidebar */}
      <aside className="w-full lg:w-72 lg:h-screen bg-white border-b lg:border-r border-slate-200 flex flex-col z-10 shadow-lg relative shrink-0">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-100/50 to-transparent pointer-events-none" />
        <div className="p-6 lg:p-8 lg:pt-10 flex lg:flex-col items-center lg:items-start justify-between lg:justify-start">
          <div className="flex items-center lg:block">
            <img src="/favicon.svg" alt="Match Logo" className="w-8 h-8 lg:w-full lg:h-auto lg:mb-6 mr-3 lg:mr-0" />
            <div>
              <h1 className="text-xl lg:text-2xl font-bold text-slate-800 tracking-tight flex items-center gap-2 lg:gap-3">
                <Monitor className="w-5 h-5 lg:w-7 lg:h-7 text-[#b1ca21]" />
                Calibrate
              </h1>
              <p className="hidden lg:block text-slate-500 text-xs mt-2 uppercase tracking-widest font-medium">Connect Four AI</p>
            </div>
          </div>
        </div>

        <nav className="px-4 space-y-1 lg:space-y-2 lg:mt-2 relative z-10 overflow-x-auto flex lg:flex-col pb-4 lg:pb-0">
          <button 
            onClick={() => navigate('/define-board')}
            className={`flex-shrink-0 lg:w-full flex items-center gap-2 lg:gap-3 px-3 py-2 lg:px-4 lg:py-3.5 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 group ${activeTab === 'define-board' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <LayoutGrid className={`w-4 h-4 lg:w-5 lg:h-5 ${activeTab === 'define-board' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            Define Board
          </button>
          
          <button 
            onClick={() => navigate('/realsense-config')}
            className={`flex-shrink-0 lg:w-full flex items-center gap-2 lg:gap-3 px-3 py-2 lg:px-4 lg:py-3.5 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 group ${activeTab === 'realsense' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Settings className={`w-4 h-4 lg:w-5 lg:h-5 ${activeTab === 'realsense' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            RealSense Calib.
          </button>

          <button 
            onClick={() => navigate('/color-calibration')}
            className={`flex-shrink-0 lg:w-full flex items-center gap-2 lg:gap-3 px-3 py-2 lg:px-4 lg:py-3.5 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 group ${activeTab === 'color-calibration' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Palette className={`w-4 h-4 lg:w-5 lg:h-5 ${activeTab === 'color-calibration' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            Color Calib.
          </button>

          <button 
            onClick={() => navigate('/detection-calibration')}
            className={`flex-shrink-0 lg:w-full flex items-center gap-2 lg:gap-3 px-3 py-2 lg:px-4 lg:py-3.5 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 group ${activeTab === 'detection-calibration' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Target className={`w-4 h-4 lg:w-5 lg:h-5 ${activeTab === 'detection-calibration' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            Detection Calib.
          </button>

          <button 
            onClick={() => navigate('/nfc-testing')}
            className={`flex-shrink-0 lg:w-full flex items-center gap-2 lg:gap-3 px-3 py-2 lg:px-4 lg:py-3.5 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 group ${activeTab === 'nfc-testing' ? 'bg-[#b1ca21]/10 text-[#8a9e19] border border-[#b1ca21]/20 shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 border border-transparent'}`}
          >
            <Nfc className={`w-4 h-4 lg:w-5 lg:h-5 ${activeTab === 'nfc-testing' ? 'text-[#b1ca21]' : 'text-slate-400 group-hover:text-slate-600'}`} />
            NFC Testing
          </button>
        </nav>
        
        <Routes>
          <Route path="/" element={<Navigate to="/define-board" replace />} />
          <Route path="*" element={null} />
        </Routes>

        <div className="hidden lg:block mt-auto p-6 relative z-10">
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
      <main className="flex-1 flex flex-col overflow-y-auto bg-slate-50/50 relative">
        <div className="p-4 lg:p-10 max-w-5xl mx-auto w-full relative z-10 pb-20">
          
          <div className={activeTab === 'nfc-testing' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">NFC Testing</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Verify the connection to the USB NFC reader and test scanning tags.</p>
              </header>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Connection Status</CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4 items-center justify-center py-8">
                    {status.nfc_connected ? (
                      <div className="flex flex-col items-center gap-3 text-emerald-600">
                        <CheckCircle2 className="w-16 h-16" />
                        <span className="text-xl font-bold">Connected</span>
                        <span className="text-sm text-slate-500">USB Device detected at /dev/ttyUSB0</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3 text-red-500">
                        <RefreshCw className="w-16 h-16 animate-spin" />
                        <span className="text-xl font-bold">Disconnected</span>
                        <span className="text-sm text-slate-500">Please plug in the USB NFC reader</span>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card className="bg-white border-slate-200 shadow-sm">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Last Scanned Tag</CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-4 items-center justify-center py-8">
                    {status.nfc_last_tag ? (
                      <div className="flex flex-col items-center gap-3">
                        <Nfc className="w-16 h-16 text-emerald-600" />
                        <span className="text-2xl font-mono font-bold text-slate-800">{status.nfc_last_tag}</span>
                        <span className="text-sm text-slate-500">Successfully scanned</span>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center gap-3 text-slate-400">
                        <Nfc className="w-16 h-16 opacity-50" />
                        <span className="text-xl font-medium">No Tag Scanned</span>
                        <span className="text-sm text-slate-400">Hold a tag against the reader</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>

          <div className={activeTab === 'define-board' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">Define Game Board</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Click on the image to set the four corners of the Connect Four grid. Once four corners are set, you can click near any corner to adjust its position.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md overflow-hidden p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img 
                    ref={imageRef}
                    src={`/frame/color?t=${sessionTime}`}
                    alt="Color Feed" 
                    className="w-full h-auto cursor-crosshair relative z-0"
                    onClick={handleImageClick}
                  />
                  <div className="absolute top-2 left-2 lg:top-4 lg:left-4 bg-white/90 text-slate-800 text-xs px-3 py-1.5 rounded-full font-medium shadow-md flex items-center gap-2 z-20">
                     <div className="w-1.5 h-1.5 rounded-full bg-[#b1ca21] animate-pulse" />
                     {status.corners?.length < 4 ? (
                      <span>Click to set corner {status.corners.length + 1}/4</span>
                    ) : (
                      <span>Click near a corner to move it</span>
                    )}
                  </div>
                </div>
              </Card>

              <div className="flex flex-col sm:flex-row gap-4 bg-white rounded-2xl border border-slate-200 shadow-sm p-4">
                <Button variant="outline" onClick={() => action('reset')} className="flex-1 bg-white border-slate-200 hover:bg-slate-50 text-slate-700 h-12">
                  <RefreshCw className="w-4 h-4 mr-2" /> Reset Corners
                </Button>
                <Button onClick={() => action('save_detection')} className="flex-1 bg-slate-800 hover:bg-slate-700 text-white shadow-md h-12">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Corners
                </Button>
              </div>

              <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                  <CardTitle className="text-lg text-slate-800">Board Geometry</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <SliderWithInput label="Hole Diameter (px)" description="The visual size of the holes mapped on the game board." value={status.hole_diameter} max={150} onChange={(v) => updateDetection('hole_diameter', v)} />
                </CardContent>
              </Card>
            </div>
          </div>

          <div className={activeTab === 'color-calibration' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">Color Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Fill the black reference columns for Player 1 and green reference columns for Player 2. Auto calibration tests image settings against every highlighted slot.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md overflow-hidden p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img src={`/frame/color?t=${sessionTime}`} alt="Color Feed" className="w-full h-auto relative z-0" />
                </div>
              </Card>

              <div className="flex flex-col sm:flex-row gap-4 p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
                <Button onClick={() => action('autocalibrate_colors')} disabled={status.corners?.length < 4 || status.is_color_autocalibrating} className="flex-1 bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md shadow-[#b1ca21]/20 h-12 transition-all">
                  <RefreshCw className={`w-4 h-4 mr-2 ${status.is_color_autocalibrating ? 'animate-spin' : ''}`} />
                  {status.is_color_autocalibrating ? 'Auto Calibrating…' : 'Auto Calibrate Colours'}
                </Button>
                <Button onClick={() => action('calibrate_colors')} disabled={status.corners?.length < 4} className="flex-1 bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md shadow-[#b1ca21]/20 h-12 transition-all">
                  <CheckCircle2 className="w-4 h-4 mr-2" /> Calibrate Colors
                </Button>
                <Button onClick={() => action('save_detection')} className="flex-1 bg-slate-800 hover:bg-slate-700 text-white shadow-md h-12">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Colors
                </Button>
              </div>

              {status.is_color_autocalibrating && (
                <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 space-y-2">
                  <div className="flex justify-between text-sm font-medium text-slate-600"><span>Testing contrast, saturation and brightness</span><span>{Math.round((status.color_autocalibrate_progress || 0) * 100)}%</span></div>
                  <div className="h-2 rounded-full bg-slate-100 overflow-hidden"><div className="h-full bg-[#b1ca21] transition-all duration-300" style={{ width: `${(status.color_autocalibrate_progress || 0) * 100}%` }} /></div>
                </div>
              )}

              {status.color_autocalibrate_result && !status.is_color_autocalibrating && (
                <p className="text-sm text-slate-600 bg-[#b1ca21]/10 border border-[#b1ca21]/20 rounded-xl px-4 py-3">
                  Auto result: {status.color_autocalibrate_result.correct}/{status.color_autocalibrate_result.total} slots correct — contrast {status.color_autocalibrate_result.contrast}, saturation {status.color_autocalibrate_result.saturation}, brightness {status.color_autocalibrate_result.brightness}.
                </p>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="bg-white border-slate-200 shadow-sm md:col-span-2">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Calibrated Colors</CardTitle>
                    <CardDescription className="text-slate-500">Colors detected for Player 1 and Player 2 tokens.</CardDescription>
                  </CardHeader>
                  <CardContent className="flex flex-col md:flex-row gap-6">
                    <div className="flex-1 p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                      <span className="font-medium text-slate-700">Player 1 (Black)</span>
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
                      <span className="font-medium text-slate-700">Player 2 (Green)</span>
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

                <Card className="bg-white border-slate-200 shadow-sm md:col-span-2">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                    <CardTitle className="text-lg text-slate-800">Image Filtering</CardTitle>
                    <CardDescription className="text-slate-500">Enhance token colors before detection.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <SliderWithInput label="Contrast" description="Adjust contrast to make token colors stand out." value={status.contrast} max={300} onChange={(v) => updateDetection('contrast', v)} />
                    <SliderWithInput label="Saturation" description="Increase color saturation to improve detection." value={status.saturation} max={300} onChange={(v) => updateDetection('saturation', v)} />
                    <SliderWithInput label="Brightness" description="Adjust brightness to compensate for room lighting." value={status.brightness} min={-100} max={100} onChange={(v) => updateDetection('brightness', v)} />
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>

          <div className={activeTab === 'detection-calibration' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">Detection Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Adjust parameters to ensure reliable token detection. The image shows detected chips or empty holes.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md overflow-hidden p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img src={`/frame/color?t=${sessionTime}`} alt="Color Feed" className="w-full h-auto relative z-0" />
                </div>
              </Card>

              <div className="flex justify-end p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
                <Button size="lg" onClick={() => action('save_detection')} className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-white shadow-md px-8">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Detection Config
                </Button>
              </div>

              <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                  <CardTitle className="text-lg text-slate-800">Detection Parameters</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <SliderWithInput 
                      label="Occupancy Threshold (Ratio)" 
                      description={<>Minimum percentage of valid pixels needed in the hole to consider it blocked by a token. <br/><span className="font-semibold">Lower values:</span> More sensitive, <span className="font-semibold">Higher values:</span> Requires a more solid reading.</>}
                      value={status.occupancy_threshold || 0.3} 
                      min={0} 
                      max={1} 
                      step={0.05} 
                      onChange={(v) => updateDetection('occupancy_threshold', v)} 
                    />
                  </div>
                  <div>
                    <SliderWithInput 
                      label="Temporal Smoothing (Frames)" 
                      description="Number of frames to consider for stability. Prevents flickering."
                      value={status.temporal_smoothing || 10} 
                      min={1} 
                      max={30} 
                      step={1} 
                      onChange={(v) => updateDetection('temporal_smoothing', v)} 
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className={activeTab === 'realsense' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">RealSense Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Configure the physical camera hardware. These settings are applied directly to the RealSense sensor.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img 
                    ref={depthRef}
                    src={`/frame/depth?t=${sessionTime}`} 
                    alt="Depth Feed" 
                    className="w-full h-auto cursor-crosshair"
                    onClick={handleDepthClick}
                  />
                  {measuredDepth !== null && (
                    <div className="absolute top-2 right-2 lg:top-4 lg:right-4 bg-white/90 text-slate-800 text-xs lg:text-sm px-3 py-1.5 lg:px-4 lg:py-2 rounded-xl font-mono shadow-md flex items-center gap-2 z-20 border border-slate-200">
                      <div className="w-1.5 h-1.5 lg:w-2 lg:h-2 rounded-full bg-[#b1ca21] animate-pulse" />
                      Depth: <span className="font-bold text-[#b1ca21]">{measuredDepth} mm</span>
                    </div>
                  )}
                </div>
              </Card>

              <div className="flex flex-col md:flex-row items-center justify-between my-4 lg:my-6 gap-4">
                <div className="hidden md:block w-32"></div> {/* Spacer to center the tabs if we want, or just let them distribute */}
                <div className="inline-flex bg-slate-100 p-1.5 rounded-2xl border border-slate-200 shadow-sm relative">
                  <button onClick={() => setRealsenseSubTab('sensor')} className={`relative z-10 px-4 py-2 lg:px-8 lg:py-2.5 text-xs lg:text-sm font-semibold rounded-xl transition-all duration-300 ${realsenseSubTab === 'sensor' ? 'text-[#8a9e19] shadow-sm bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'}`}>
                    Sensor Control
                  </button>
                  <button onClick={() => setRealsenseSubTab('filtering')} className={`relative z-10 px-4 py-2 lg:px-8 lg:py-2.5 text-xs lg:text-sm font-semibold rounded-xl transition-all duration-300 ${realsenseSubTab === 'filtering' ? 'text-[#8a9e19] shadow-sm bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'}`}>
                    Depth Filtering
                  </button>
                </div>
                <Button onClick={() => action('save_realsense')} disabled={status.autocalibrate_state !== 0} className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-white shadow-md">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Profile
                </Button>
              </div>

              {realsenseSubTab === 'sensor' && (
                <div className="space-y-6">
                  <Card className="bg-white border-slate-200 shadow-sm p-4 lg:p-6">
                    <div className="flex flex-col justify-between gap-4">
                      {status.autocalibrate_state === 0 && (
                        <div className="flex flex-col gap-3">
                          <div className="text-sm font-semibold text-slate-700">Auto Calibration Options</div>
                          <div className="flex flex-wrap items-center gap-4">
                            <Button size="lg" onClick={() => action('autocalibrate_single', advancedSettings)} disabled={status.corners?.length < 4} className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md px-6 transition-all">
                              <RefreshCw className="w-4 h-4 mr-2" /> Quick Calibrate
                            </Button>
                            <Button size="lg" onClick={() => action('autocalibrate_step1', advancedSettings)} disabled={status.corners?.length < 4} variant="outline" className="border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21]/10 px-6 transition-all">
                              Thorough Calibrate (2-Step)
                            </Button>
                          </div>
                          
                          <div className="mt-4 p-4 border border-slate-200 rounded-lg bg-slate-50/50 space-y-4">
                            <div className="flex items-center justify-between mb-2">
                              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Advanced Sweep Parameters</div>
                              {(() => {
                                const expSteps = Math.max(1, Math.floor((advancedSettings.exp_max - advancedSettings.exp_min) / (advancedSettings.exp_step || 1)) + 1)
                                const gainSteps = Math.max(1, Math.floor((advancedSettings.gain_max - advancedSettings.gain_min) / (advancedSettings.gain_step || 1)) + 1)
                                const laserSteps = Math.max(1, Math.floor((advancedSettings.laser_max - advancedSettings.laser_min) / (advancedSettings.laser_step || 1)) + 1)
                                const combinations = expSteps * gainSteps * laserSteps
                                const estimatedTimeSeconds = combinations * (advancedSettings.duration + 0.6)
                                const timeStr = estimatedTimeSeconds > 60 ? `${Math.floor(estimatedTimeSeconds / 60)}m ${Math.round(estimatedTimeSeconds % 60)}s` : `${Math.round(estimatedTimeSeconds)}s`
                                return (
                                  <div className="text-xs font-medium bg-[#b1ca21]/20 text-[#8a9e19] px-2 py-1 rounded-md">
                                    Est. Time: {timeStr} ({combinations} combos)
                                  </div>
                                )
                              })()}
                            </div>
                            <div className="mb-4">
                              <SliderWithInput label="Recording Duration per Setting (s)" description="How long to record depth data for each combination of settings to evaluate stability (default: 3s)." value={advancedSettings.duration} min={0.5} max={10} step={0.5} onChange={(v) => setAdvancedSettings({...advancedSettings, duration: v})} />
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Exposure Range</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.exp_min} onChange={e => setAdvancedSettings({...advancedSettings, exp_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.exp_max} onChange={e => setAdvancedSettings({...advancedSettings, exp_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Step</Label><Input type="number" value={advancedSettings.exp_step} onChange={e => setAdvancedSettings({...advancedSettings, exp_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Gain Range</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.gain_min} onChange={e => setAdvancedSettings({...advancedSettings, gain_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.gain_max} onChange={e => setAdvancedSettings({...advancedSettings, gain_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Step</Label><Input type="number" value={advancedSettings.gain_step} onChange={e => setAdvancedSettings({...advancedSettings, gain_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Laser Range</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.laser_min} onChange={e => setAdvancedSettings({...advancedSettings, laser_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.laser_max} onChange={e => setAdvancedSettings({...advancedSettings, laser_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Step</Label><Input type="number" value={advancedSettings.laser_step} onChange={e => setAdvancedSettings({...advancedSettings, laser_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {status.autocalibrate_state === 0 && status.autocalibrate_results && status.autocalibrate_results.length > 0 && (
                        <div className="mt-6 border-t border-slate-100 pt-6">
                          <h4 className="text-sm font-semibold text-slate-700 mb-3">Top Calibration Results</h4>
                          <div className="bg-slate-50 border border-slate-200 rounded-lg overflow-hidden">
                            <div className="max-h-64 overflow-y-auto">
                              <table className="w-full text-left text-xs relative">
                                <thead className="bg-slate-100 text-slate-600 sticky top-0 shadow-sm">
                                  <tr>
                                    <th className="px-3 py-2 font-medium">Rank</th>
                                    <th className="px-3 py-2 font-medium">Exp</th>
                                    <th className="px-3 py-2 font-medium">Gain</th>
                                    <th className="px-3 py-2 font-medium">Laser</th>
                                    <th className="px-3 py-2 font-medium">Score</th>
                                    <th className="px-3 py-2 font-medium text-right">Action</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-200">
                                  {status.autocalibrate_results.map((res: any, idx: number) => (
                                    <tr key={idx} className="hover:bg-white transition-colors">
                                      <td className="px-3 py-2 font-mono text-slate-500">#{idx + 1}</td>
                                      <td className="px-3 py-2">{res.exposure}</td>
                                      <td className="px-3 py-2">{res.gain}</td>
                                      <td className="px-3 py-2">{res.laser}</td>
                                      <td className="px-3 py-2">
                                        <span className="font-semibold text-emerald-600">{res.score}/42</span>
                                        <span className="text-slate-400 ml-1">(var: {res.var.toFixed(1)})</span>
                                      </td>
                                      <td className="px-3 py-2 text-right">
                                        <Button 
                                          size="sm" 
                                          variant="outline" 
                                          className="h-7 text-xs border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21] hover:text-white"
                                          onClick={() => {
                                          fetch('/api/update_realsense', {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({
                                              exposure: res.exposure,
                                              gain: res.gain,
                                              laser_power: res.laser
                                            })
                                          }).then(() => setPendingOverrides({}))
                                        }}
                                        >
                                          Use
                                        </Button>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
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
                            <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Cancel</Button>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-[#b1ca21] h-2 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                          </div>
                        </div>
                      )}

                      {status.autocalibrate_state === 1 && (
                        <div className="flex flex-col gap-3">
                          <div className="flex items-center gap-2">
                            <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Step 1: Scanning Empty Board...
                            </Button>
                            <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Cancel</Button>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-[#b1ca21] h-2 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                          </div>
                        </div>
                      )}

                      {status.autocalibrate_state === 2 && (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-4">
                            <Button size="lg" onClick={() => action('autocalibrate_step2')} className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white px-6 animate-pulse">
                              <RefreshCw className="w-4 h-4 mr-2" /> Scan Filled Board (Step 2)
                            </Button>
                            <Button variant="outline" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Cancel</Button>
                          </div>
                        </div>
                      )}

                      {status.autocalibrate_state === 3 && (
                        <div className="flex flex-col gap-3">
                          <div className="flex items-center gap-2">
                            <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> Step 2: Scanning Filled Board...
                            </Button>
                            <Button variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Cancel</Button>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-[#b1ca21] h-2 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>

                  <Card className="bg-white border-slate-200 shadow-sm">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                      <CardTitle className="text-lg text-slate-800">Manual Overrides</CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <SliderWithInput label="Exposure" description="Camera exposure time (lower values reduce blur but darken image)." value={pendingOverrides.exposure !== undefined ? pendingOverrides.exposure : status.exposure} min={1} max={10000} step={50} onChange={(v) => setPendingOverrides({...pendingOverrides, exposure: v})} />
                        <SliderWithInput label="Gain" description="Sensor signal gain (amplifies signal but increases noise)." value={pendingOverrides.gain !== undefined ? pendingOverrides.gain : status.gain} min={16} max={248} onChange={(v) => setPendingOverrides({...pendingOverrides, gain: v})} />
                        <SliderWithInput label="Laser Power" description="Intensity of the IR projector for depth estimation." value={pendingOverrides.laser_power !== undefined ? pendingOverrides.laser_power : status.laser_power} min={0} max={360} onChange={(v) => setPendingOverrides({...pendingOverrides, laser_power: v})} />
                        <SliderWithInput label="Visual Preset" description="Hardware optimization preset (3 = High Accuracy)." value={pendingOverrides.visual_preset !== undefined ? pendingOverrides.visual_preset : status.visual_preset} min={0} max={5} onChange={(v) => setPendingOverrides({...pendingOverrides, visual_preset: v})} />
                      </div>
                      <div className="flex justify-end mt-2">
                        <Button 
                          className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white disabled:opacity-50"
                          disabled={Object.keys(pendingOverrides).length === 0}
                          onClick={() => {
                            if (Object.keys(pendingOverrides).length === 0) return;
                            fetch('/api/update_realsense', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify(pendingOverrides)
                            }).then(() => setPendingOverrides({}))
                          }}
                        >
                          Apply Manual Overrides
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {realsenseSubTab === 'filtering' && (
                <div className="space-y-6">
                  <Card className="bg-white border-slate-200 shadow-sm flex flex-col">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                      <CardTitle className="text-lg text-slate-800">Depth Filtering</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <SliderWithInput label="Min Depth (mm)" description="Ignore all pixels closer than this distance." value={status.min_depth} max={5000} step={10} onChange={(v) => updateRealSense('min_depth', v)} />
                      <SliderWithInput label="Max Depth (mm)" description="Ignore all pixels further than this distance." value={status.max_depth} max={5000} step={10} onChange={(v) => updateRealSense('max_depth', v)} />
                      <div className="mt-6 p-5 bg-white rounded-xl border border-slate-200 flex items-center justify-between shadow-sm">
                        <Label className="text-slate-700 font-medium tracking-wide text-sm cursor-pointer" htmlFor="emitter-toggle">
                          Emitter Enabled
                        </Label>
                        <input id="emitter-toggle" type="checkbox" checked={status.emitter === 1} onChange={(e) => updateRealSense('emitter', e.target.checked ? 1 : 0)} className="w-5 h-5 rounded" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}



            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
