import { useEffect, useState, useRef, type MouseEvent } from 'react'
import { useLocation, useNavigate, Routes, Route, Navigate } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './components/ui/card'
import { Button } from './components/ui/button'
import { Slider } from './components/ui/slider'
import { Label } from './components/ui/label'
import { Input } from './components/ui/input'
import { Settings, Monitor, RefreshCw, Save, CheckCircle2, LayoutGrid, Palette, Target, Nfc } from 'lucide-react'

function SliderWithInput({ label, description, value, min = 0, max = 100, step = 1, onChange, disabled = false }: { label: string, description?: React.ReactNode, value: number, min?: number, max?: number, step?: number, onChange: (v: number) => void, disabled?: boolean }) {
  return (
    <div className={`space-y-4 p-5 bg-white rounded-xl border border-slate-200 transition-all duration-300 shadow-sm ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-[#b1ca21]/50 hover:bg-slate-50'}`}>
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
          disabled={disabled}
          className="flex-1 cursor-pointer disabled:cursor-not-allowed [&_[role=slider]]:bg-[#b1ca21] [&_[role=slider]]:border-[#b1ca21] [&_[data-orientation=horizontal]>span:first-child]:bg-slate-200 [&_[data-orientation=horizontal]>span:first-child>span]:bg-[#b1ca21]"
        />
        <Input 
          type="number" 
          value={value} 
          min={min}
          max={max}
          disabled={disabled}
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
    
  const [realsenseSubTab, setRealsenseSubTab] = useState<'automatic' | 'manual' | 'filtering'>('automatic')
  const [measuredDepth, setMeasuredDepth] = useState<number | null>(null)
  const [advancedSettings, setAdvancedSettings] = useState({
    exp_min: 1000, exp_max: 8000, exp_step: 1500,
    gain_min: 16, gain_max: 128, gain_step: 24,
    laser_min: 150, laser_max: 360, laser_step: 75,
    duration: 3.0
  })
  const [pendingOverrides, setPendingOverrides] = useState<any>({})
  const [pendingColorCamera, setPendingColorCamera] = useState<any>({})
  const [pendingColorImage, setPendingColorImage] = useState<any>({})
  const [colorCalibrationTab, setColorCalibrationTab] = useState<'automatic' | 'manual'>('automatic')
  const [colorPrecision, setColorPrecision] = useState<'fast' | 'standard' | 'thorough'>('standard')
  const [saveFeedback, setSaveFeedback] = useState<{ type: 'success' | 'error', message: string } | null>(null)

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

  const applyManualColorSettings = async () => {
    if (Object.keys(pendingColorCamera).length > 0) {
      await fetch('/api/update_realsense', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pendingColorCamera) })
      setPendingColorCamera({})
    }
    if (Object.keys(pendingColorImage).length > 0) {
      await fetch('/api/update_detection', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pendingColorImage) })
      setPendingColorImage({})
    }
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

  const saveConfiguration = async (endpoint: string, description: string) => {
    try {
      const response = await fetch(`/api/${endpoint}`, { method: 'POST' })
      if (!response.ok) {
        setSaveFeedback({ type: 'error', message: `${description} konnten nicht gespeichert werden.` })
      } else {
        setSaveFeedback({ type: 'success', message: `${description} wurden dauerhaft gespeichert.` })
      }
    } catch {
      setSaveFeedback({ type: 'error', message: `${description} konnten nicht gespeichert werden.` })
    } finally {
      fetchStatus()
      window.setTimeout(() => setSaveFeedback(null), 5000)
    }
  }

  if (!status) return <div className="min-h-screen bg-slate-50 flex items-center justify-center text-[#b1ca21] animate-pulse text-lg tracking-widest font-light">INITIALIZING...</div>

  const manualControlsLocked = status.autocalibrate_state !== 0 || status.is_color_capturing || status.is_color_autocalibrating

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
        {saveFeedback && (
          <div role="status" className={`fixed right-4 bottom-4 z-50 max-w-md rounded-xl border px-4 py-3 shadow-lg text-sm font-medium ${saveFeedback.type === 'success' ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-red-50 border-red-200 text-red-800'}`}>
            {saveFeedback.type === 'success' ? '✓ ' : '✕ '}{saveFeedback.message}
          </div>
        )}
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
                <Button variant="outline" disabled={manualControlsLocked} onClick={() => action('reset')} className="flex-1 bg-white border-slate-200 hover:bg-slate-50 text-slate-700 h-12">
                  <RefreshCw className="w-4 h-4 mr-2" /> Reset Corners
                </Button>
                <Button onClick={() => saveConfiguration('save_detection', 'Die Erkennungseinstellungen')} disabled={manualControlsLocked} className="flex-1 bg-slate-800 hover:bg-slate-700 text-white shadow-md h-12">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save Corners
                </Button>
              </div>

              <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                  <CardTitle className="text-lg text-slate-800">Board Geometry</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <SliderWithInput label="Hole Diameter (px)" description="The visual size of the holes mapped on the game board." value={status.hole_diameter} max={150} disabled={manualControlsLocked} onChange={(v) => updateDetection('hole_diameter', v)} />
                </CardContent>
              </Card>
            </div>
          </div>

          <div className={activeTab === 'color-calibration' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">Colour Calibration</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-3xl">The guided flow samples legal board positions at low, medium and full occupancy. Put Player 1 (black) in columns 1, 3, 5 and 7; put Player 2 (green) in columns 2, 4 and 6. The live feed marks only the slots needed for the current step.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md overflow-hidden p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img src={`/frame/color?t=${sessionTime}`} alt="Color Feed" className="w-full h-auto relative z-0" />
                </div>
              </Card>

              <div className="flex flex-col sm:flex-row gap-4 p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
                <Button onClick={() => saveConfiguration('save_realsense', 'Die RealSense-Einstellungen')} disabled={manualControlsLocked} className="flex-1 bg-slate-800 hover:bg-slate-700 text-white h-12">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save RGB settings to config
                </Button>
                <Button onClick={() => saveConfiguration('save_detection', 'Die Erkennungseinstellungen')} disabled={manualControlsLocked} className="flex-1 bg-slate-800 hover:bg-slate-700 text-white h-12">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Save calibrated colours
                </Button>
              </div>

              <Card className="bg-white border-slate-200 shadow-sm">
                <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                  <CardTitle className="text-lg text-slate-800">Currently calibrated colours</CardTitle>
                  <CardDescription className="text-slate-500">These are the colour references currently used by detection.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col md:flex-row gap-6">
                  <div className="flex-1 p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between"><span className="font-medium text-slate-700">Player 1 (Black)</span>{status.player1_color ? <div className="flex items-center gap-3"><span className="text-xs font-mono text-slate-500">BGR: [{status.player1_color.join(', ')}]</span><div className="w-8 h-8 rounded-full border-2 border-slate-300" style={{ backgroundColor: `rgb(${status.player1_color[2]}, ${status.player1_color[1]}, ${status.player1_color[0]})` }} /></div> : <span className="text-xs text-slate-400 italic">Not calibrated</span>}</div>
                  <div className="flex-1 p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between"><span className="font-medium text-slate-700">Player 2 (Green)</span>{status.player2_color ? <div className="flex items-center gap-3"><span className="text-xs font-mono text-slate-500">BGR: [{status.player2_color.join(', ')}]</span><div className="w-8 h-8 rounded-full border-2 border-slate-300" style={{ backgroundColor: `rgb(${status.player2_color[2]}, ${status.player2_color[1]}, ${status.player2_color[0]})` }} /></div> : <span className="text-xs text-slate-400 italic">Not calibrated</span>}</div>
                </CardContent>
              </Card>

              <div className="pt-2">
                <div className="inline-flex bg-slate-100 p-1.5 rounded-t-2xl rounded-br-none border border-slate-200 border-b-0 shadow-sm relative z-10">
                  <button onClick={() => setColorCalibrationTab('automatic')} className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${colorCalibrationTab === 'automatic' ? 'text-[#8a9e19] bg-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Automatic calibration</button>
                  <button onClick={() => setColorCalibrationTab('manual')} className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${colorCalibrationTab === 'manual' ? 'text-[#8a9e19] bg-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Manual calibration</button>
                </div>
                <div>

              {colorCalibrationTab === 'automatic' && <div className="space-y-6">
                <Card className="bg-white border-slate-200 shadow-sm rounded-tl-none">
                  <CardHeader className="pb-4 border-b border-slate-100 mb-4"><CardTitle className="text-lg text-slate-800">Automatic calibration precision</CardTitle><CardDescription className="text-slate-500">Automatic calibration tests fixed RGB exposure/gain settings, then locks the selected RGB setting. RGB auto-exposure is never left enabled.</CardDescription></CardHeader>
                  <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-3">
                    {[
                      ['fast', 'Fast', '2 layouts; 6 RGB settings. About 5 seconds per automatic capture.'],
                      ['standard', 'Standard', '3 layouts; 10 RGB settings. About 8 seconds per automatic capture.'],
                      ['thorough', 'Thorough', '3 layouts; 21 RGB settings. About 20 seconds per automatic capture.'],
                    ].map(([value, title, description]) => <button key={value} type="button" disabled={manualControlsLocked || status.is_color_capturing || status.is_color_autocalibrating} onClick={() => setColorPrecision(value as 'fast' | 'standard' | 'thorough')} className={`text-left rounded-xl border p-4 transition-all ${colorPrecision === value ? 'border-[#b1ca21] bg-[#b1ca21]/10 ring-1 ring-[#b1ca21]/30' : 'border-slate-200 hover:border-slate-300'} disabled:opacity-50`}><div className="font-semibold text-slate-800">{title}</div><p className="mt-1 text-xs leading-relaxed text-slate-500">{description}</p></button>)}
                  </CardContent>
                </Card>
                <div className="flex flex-col sm:flex-row gap-4 p-4 bg-white rounded-2xl border border-slate-200 shadow-sm">
                  <Button onClick={() => action('color_calibration/start', { precision: colorPrecision })} disabled={manualControlsLocked || status.corners?.length < 4 || status.is_color_capturing || status.is_color_autocalibrating} className="flex-1 bg-[#b1ca21] hover:bg-[#a0b51e] text-white h-12"><RefreshCw className="w-4 h-4 mr-2" /> Start guided calibration</Button>
                  <Button onClick={() => action('color_calibration/capture')} disabled={manualControlsLocked || status.color_calibration_stage_rows === null || status.color_calibration_stage_rows === undefined || status.is_color_capturing || status.is_color_autocalibrating} className="flex-1 bg-[#b1ca21] hover:bg-[#a0b51e] text-white h-12"><CheckCircle2 className={`w-4 h-4 mr-2 ${status.is_color_capturing ? 'animate-pulse' : ''}`} />{status.is_color_capturing ? 'Capturing…' : 'Capture this layout'}</Button>
                </div>
                {status.color_calibration_stage_rows !== null && status.color_calibration_stage_rows !== undefined && !status.is_color_autocalibrating && <p className="text-sm text-slate-600 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3">Step {(status.color_calibration_stage_index || 0) + 1} of {status.color_calibration_stage_count}: fill the bottom {status.color_calibration_stage_rows} row{status.color_calibration_stage_rows === 1 ? '' : 's'}, then capture this layout.</p>}
                {status.is_color_capturing && <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 space-y-2"><div className="flex justify-between text-sm font-medium text-slate-600"><span>Capturing RGB exposure/gain sweep{status.color_calibration_active_rgb_setting ? ` — exposure ${status.color_calibration_active_rgb_setting.exposure}, gain ${status.color_calibration_active_rgb_setting.gain}` : ''}</span><span>{Math.round((status.color_calibration_capture_progress || 0) * 100)}% · ~{Math.ceil(status.color_calibration_eta_seconds || 0)} s remaining</span></div><div className="h-2 rounded-full bg-slate-100 overflow-hidden"><div className="h-full bg-[#b1ca21]" style={{ width: `${(status.color_calibration_capture_progress || 0) * 100}%` }} /></div></div>}
                {status.is_color_autocalibrating && <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-4 space-y-2"><div className="flex justify-between text-sm font-medium text-slate-600"><span>Jointly selecting RGB exposure, gain and image filtering</span><span>{Math.round((status.color_autocalibrate_progress || 0) * 100)}% · ~{Math.ceil(status.color_calibration_eta_seconds || 0)} s remaining</span></div><div className="h-2 rounded-full bg-slate-100 overflow-hidden"><div className="h-full bg-[#b1ca21]" style={{ width: `${(status.color_autocalibrate_progress || 0) * 100}%` }} /></div></div>}
                {status.color_autocalibrate_result && !status.is_color_autocalibrating && <p className="text-sm text-slate-600 bg-[#b1ca21]/10 border border-[#b1ca21]/20 rounded-xl px-4 py-3">Auto result: {status.color_autocalibrate_result.accuracy}% stage-balanced accuracy — RGB exposure {status.color_autocalibrate_result.rgb_exposure}, RGB gain {status.color_autocalibrate_result.rgb_gain}, contrast {status.color_autocalibrate_result.contrast}, saturation {status.color_autocalibrate_result.saturation}, brightness {status.color_autocalibrate_result.brightness}. Choose another ranked result below before saving if you prefer.</p>}
                {status.color_autocalibrate_results?.length > 0 && !status.is_color_autocalibrating && <Card className="bg-white border-slate-200 shadow-sm overflow-hidden"><CardHeader className="pb-4 border-b border-slate-100"><CardTitle className="text-lg text-slate-800">Automatic calibration results</CardTitle><CardDescription className="text-slate-500">Try an alternative before using the save buttons above. The selected result is applied temporarily; saving remains explicit.</CardDescription></CardHeader><CardContent className="p-0 overflow-x-auto"><table className="w-full text-left text-xs"><thead className="bg-slate-50 text-slate-500"><tr><th className="px-3 py-3">#</th><th className="px-3 py-3">Accuracy</th><th className="px-3 py-3">Exposure</th><th className="px-3 py-3">Gain</th><th className="px-3 py-3">Contrast</th><th className="px-3 py-3">Saturation</th><th className="px-3 py-3">Brightness</th><th className="px-3 py-3 text-right">Aktion</th></tr></thead><tbody className="divide-y divide-slate-100">{status.color_autocalibrate_results.map((result: any, index: number) => <tr key={index} className="hover:bg-slate-50"><td className="px-3 py-3 font-medium text-slate-700">{index + 1}</td><td className="px-3 py-3">{result.accuracy}%<span className="text-slate-400"> · margin {result.margin}</span></td><td className="px-3 py-3">{result.rgb_exposure} µs</td><td className="px-3 py-3">{result.rgb_gain}</td><td className="px-3 py-3">{result.contrast}</td><td className="px-3 py-3">{result.saturation}</td><td className="px-3 py-3">{result.brightness}</td><td className="px-3 py-3 text-right"><Button size="sm" variant="outline" onClick={() => action('color_calibration/use_result', { index })} className="border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21]/10">Use</Button></td></tr>)}</tbody></table></CardContent></Card>}
              </div>}

              {colorCalibrationTab === 'manual' && <div className="space-y-6">
                <Card className="bg-white border-slate-200 shadow-sm rounded-tl-none"><CardHeader className="pb-4 border-b border-slate-100 mb-4"><CardTitle className="text-lg text-slate-800">Manual RGB and image settings</CardTitle><CardDescription className={status.color_sensor_available && status.color_exposure_supported ? "text-slate-500" : "text-amber-600"}>{status.color_sensor_available && status.color_exposure_supported ? 'Slider Wechsel are pending. They are not applied until you press Apply Manual Settings.' : 'RGB manual-exposure support was not detected; connect a RealSense colour sensor before colour calibration.'}</CardDescription></CardHeader>
                  <CardContent className="space-y-4"><div className="p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-start justify-between gap-4"><div><Label className="text-slate-700 font-medium text-sm">Lock RGB exposure for colour detection</Label><p className="text-xs text-slate-500 mt-1">Disable RGB auto-exposure before using a manual colour reference.</p></div><input type="checkbox" disabled={manualControlsLocked} checked={(pendingColorCamera.color_auto_exposure !== undefined ? pendingColorCamera.color_auto_exposure : status.color_auto_exposure) === 0} onChange={(e) => setPendingColorCamera({...pendingColorCamera, color_auto_exposure: e.target.checked ? 0 : 1})} className="w-5 h-5 rounded accent-[#b1ca21]" /></div>
                  {(pendingColorCamera.color_auto_exposure !== undefined ? pendingColorCamera.color_auto_exposure : status.color_auto_exposure) === 0 && <div className="grid grid-cols-1 md:grid-cols-2 gap-4"><SliderWithInput label="RGB Exposure (µs)" description="Belichtungszeit in Mikrosekunden: 1.000 = 1 ms, 5.000 = 5 ms. Höher macht das Bild heller, kann aber helle Bereiche ausbrennen lassen." value={pendingColorCamera.color_exposure !== undefined ? pendingColorCamera.color_exposure : status.color_exposure} min={41} max={10000} step={50} disabled={manualControlsLocked} onChange={(v) => setPendingColorCamera({...pendingColorCamera, color_exposure: v})} /><SliderWithInput label="RGB Gain (Kamerawert)" description="Einheitenloser Sensor-Verstärkungswert. 16 ist der niedrigste Wert; höhere Werte hellen auf, erhöhen aber Bildrauschen und können die Farben verfälschen." value={pendingColorCamera.color_gain !== undefined ? pendingColorCamera.color_gain : status.color_gain} min={16} max={248} disabled={manualControlsLocked} onChange={(v) => setPendingColorCamera({...pendingColorCamera, color_gain: v})} /></div>}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4"><SliderWithInput label="Contrast" description="Pending image-processing contrast." value={pendingColorImage.contrast !== undefined ? pendingColorImage.contrast : status.contrast} max={300} disabled={manualControlsLocked} onChange={(v) => setPendingColorImage({...pendingColorImage, contrast: v})} /><SliderWithInput label="Saturation" description="Pending image-processing saturation." value={pendingColorImage.saturation !== undefined ? pendingColorImage.saturation : status.saturation} max={300} disabled={manualControlsLocked} onChange={(v) => setPendingColorImage({...pendingColorImage, saturation: v})} /><SliderWithInput label="Brightness" description="Pending image-processing brightness." value={pendingColorImage.brightness !== undefined ? pendingColorImage.brightness : status.brightness} min={-100} max={100} disabled={manualControlsLocked} onChange={(v) => setPendingColorImage({...pendingColorImage, brightness: v})} /></div>
                  <div className="flex flex-col sm:flex-row justify-end gap-4"><Button disabled={manualControlsLocked || (Object.keys(pendingColorCamera).length === 0 && Object.keys(pendingColorImage).length === 0)} onClick={applyManualColorSettings} className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white">Apply Manual Settings</Button><Button onClick={() => action('calibrate_colors')} disabled={manualControlsLocked || status.corners?.length < 4} variant="outline" className="border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21]/10">Calibrate full board</Button></div></CardContent></Card>
              </div>}
                </div>
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
                <Button size="lg" onClick={() => saveConfiguration('save_detection', 'Die Erkennungseinstellungen')} disabled={manualControlsLocked} className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-white shadow-md px-8">
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
                      disabled={manualControlsLocked} onChange={(v) => updateDetection('occupancy_threshold', v)}
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
                      disabled={manualControlsLocked} onChange={(v) => updateDetection('temporal_smoothing', v)}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className={activeTab === 'realsense' ? 'block' : 'hidden'}>
            <div className="space-y-6 lg:space-y-8">
              <header className="mb-4 lg:mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-slate-800 tracking-tight">RealSense-Kalibrierung</h2>
                <p className="text-slate-500 mt-2 text-sm max-w-2xl min-h-[40px]">Konfiguriere die Tiefen-Kamera. Diese Einstellungen werden direkt auf den RealSense-Sensor angewendet.</p>
              </header>

              <Card className="bg-white border-slate-200 shadow-md p-2">
                <div className="relative group rounded-lg overflow-hidden border border-slate-100">
                  <img 
                    ref={depthRef}
                    src={`/frame/depth?t=${sessionTime}`} 
                    alt="Tiefenbild"
                    className="w-full h-auto cursor-crosshair"
                    onClick={handleDepthClick}
                  />
                  {measuredDepth !== null && (
                    <div className="absolute top-2 right-2 lg:top-4 lg:right-4 bg-white/90 text-slate-800 text-xs lg:text-sm px-3 py-1.5 lg:px-4 lg:py-2 rounded-xl font-mono shadow-md flex items-center gap-2 z-20 border border-slate-200">
                      <div className="w-1.5 h-1.5 lg:w-2 lg:h-2 rounded-full bg-[#b1ca21] animate-pulse" />
                      Tiefe: <span className="font-bold text-[#b1ca21]">{measuredDepth} mm</span>
                    </div>
                  )}
                </div>
              </Card>

              <div className="flex justify-start">
                <Button title="Speichert die aktuellen Tiefen-Kameraeinstellungen dauerhaft in der RealSense-Konfiguration. Sie werden beim nächsten Start wieder geladen." onClick={() => saveConfiguration('save_realsense', 'Die RealSense-Einstellungen')} disabled={status.autocalibrate_state !== 0} className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-white shadow-md">
                  <Save className="w-4 h-4 mr-2 text-[#b1ca21]" /> Tiefen-Kameraeinstellungen dauerhaft speichern
                </Button>
              </div>

              <div className="pt-2">
                <div className="inline-flex bg-slate-100 p-1.5 rounded-t-2xl rounded-br-none border border-slate-200 border-b-0 shadow-sm relative z-10">
                  <button title="Automatische Suche nach stabilen Tiefensensor-Einstellungen." onClick={() => setRealsenseSubTab('automatic')} className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${realsenseSubTab === 'automatic' ? 'text-[#8a9e19] bg-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Automatisch</button>
                  <button title="Belichtung, Verstärkung, Laserleistung und Hardware-Voreinstellung selbst setzen." onClick={() => setRealsenseSubTab('manual')} className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${realsenseSubTab === 'manual' ? 'text-[#8a9e19] bg-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Manuell</button>
                  <button title="Grenzen für gültige Tiefenwerte und den Infrarot-Emitter einstellen." onClick={() => setRealsenseSubTab('filtering')} className={`px-5 py-2.5 text-sm font-semibold rounded-xl transition-all ${realsenseSubTab === 'filtering' ? 'text-[#8a9e19] bg-white shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Tiefenfilter</button>
                </div>
                <div>

              {realsenseSubTab === 'automatic' && (
                <div className="space-y-6">
                  <Card className="bg-white border-slate-200 shadow-sm rounded-tl-none p-4 lg:p-6">
                    <div className="flex flex-col justify-between gap-4">
                      {status.autocalibrate_state === 0 && (
                        <div className="flex flex-col gap-3">
                          <div className="text-sm font-semibold text-slate-700">Automatische Tiefenkalibrierung</div>
                          <div className="flex flex-wrap items-center gap-4">
                            <Button title="Prüft die gewählten Tiefensensor-Einstellungen am leeren Feld und wählt die beste schnelle Einstellung aus." size="lg" onClick={() => action('autocalibrate_single', advancedSettings)} disabled={manualControlsLocked || status.corners?.length < 4} className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white shadow-md px-6 transition-all">
                              <RefreshCw className="w-4 h-4 mr-2" /> Schnell kalibrieren
                            </Button>
                            <Button title="Misst zuerst das leere und danach das vollständig gefüllte Feld. Liefert die verlässlichste Tiefenkalibrierung." size="lg" onClick={() => action('autocalibrate_step1', advancedSettings)} disabled={manualControlsLocked || status.corners?.length < 4} variant="outline" className="border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21]/10 px-6 transition-all">
                              Gründlich kalibrieren (2 Schritte)
                            </Button>
                          </div>
                          
                          <div className="mt-4 p-4 border border-slate-200 rounded-lg bg-slate-50/50 space-y-4">
                            <div className="flex items-center justify-between mb-2">
                              <div className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Erweiterte Suchparameter</div>
                              {(() => {
                                const expSteps = Math.max(1, Math.floor((advancedSettings.exp_max - advancedSettings.exp_min) / (advancedSettings.exp_step || 1)) + 1)
                                const gainSteps = Math.max(1, Math.floor((advancedSettings.gain_max - advancedSettings.gain_min) / (advancedSettings.gain_step || 1)) + 1)
                                const laserSteps = Math.max(1, Math.floor((advancedSettings.laser_max - advancedSettings.laser_min) / (advancedSettings.laser_step || 1)) + 1)
                                const combinations = expSteps * gainSteps * laserSteps
                                const estimatedTimeSeconds = combinations * (advancedSettings.duration + 0.6)
                                const timeStr = estimatedTimeSeconds > 60 ? `${Math.floor(estimatedTimeSeconds / 60)}m ${Math.round(estimatedTimeSeconds % 60)}s` : `${Math.round(estimatedTimeSeconds)}s`
                                return (
                                  <div className="text-xs font-medium bg-[#b1ca21]/20 text-[#8a9e19] px-2 py-1 rounded-md">
                                    Geschätzte Dauer: {timeStr} ({combinations} Kombinationen)
                                  </div>
                                )
                              })()}
                            </div>
                            <div className="mb-4">
                              <SliderWithInput label="Aufnahmedauer je Einstellung (s)" description="So lange werden Tiefendaten je Einstellung aufgezeichnet, um die Stabilität zu bewerten (Standard: 3 s)." value={advancedSettings.duration} min={0.5} max={10} step={0.5} disabled={manualControlsLocked} onChange={(v) => setAdvancedSettings({...advancedSettings, duration: v})} />
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Belichtungsbereich</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.exp_min} onChange={e => setAdvancedSettings({...advancedSettings, exp_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.exp_max} onChange={e => setAdvancedSettings({...advancedSettings, exp_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Schritt</Label><Input type="number" value={advancedSettings.exp_step} onChange={e => setAdvancedSettings({...advancedSettings, exp_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Gain-Bereich</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.gain_min} onChange={e => setAdvancedSettings({...advancedSettings, gain_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.gain_max} onChange={e => setAdvancedSettings({...advancedSettings, gain_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Schritt</Label><Input type="number" value={advancedSettings.gain_step} onChange={e => setAdvancedSettings({...advancedSettings, gain_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                              <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                                <Label className="text-sm font-semibold text-slate-700 mb-3 block">Laserleistungsbereich</Label>
                                <div className="grid grid-cols-3 gap-2">
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Min</Label><Input type="number" value={advancedSettings.laser_min} onChange={e => setAdvancedSettings({...advancedSettings, laser_min: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Max</Label><Input type="number" value={advancedSettings.laser_max} onChange={e => setAdvancedSettings({...advancedSettings, laser_max: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                  <div><Label className="text-[10px] text-slate-400 uppercase">Schritt</Label><Input type="number" value={advancedSettings.laser_step} onChange={e => setAdvancedSettings({...advancedSettings, laser_step: Number(e.target.value)})} className="h-8 text-xs font-mono" /></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {status.autocalibrate_state === 0 && status.autocalibrate_results && status.autocalibrate_results.length > 0 && (
                        <div className="mt-6 border-t border-slate-100 pt-6">
                          <h4 className="text-sm font-semibold text-slate-700 mb-1">Beste Kalibrierungsergebnisse</h4>
                          <p className="text-xs text-slate-500 mb-3">Sortiert nach Leistung aus rohen Bildern mit kreisrunder ROI – ohne zeitliche Glättung. 100/100 bedeutet: alle erwarteten Zustände korrekt und kein Flickern. Bei der Messabdeckung ist „Leer P95“ kleiner besser, „Gefüllt P05“ größer besser. „Fehler“ sind falsche Schließungen im leeren sowie falsche Öffnungen im gefüllten Feld.</p>
                          <div className="bg-slate-50 border border-slate-200 rounded-lg overflow-hidden">
                            <div className="max-h-72 overflow-y-auto overflow-x-auto">
                              <table className="w-full min-w-[900px] text-left text-xs relative">
                                <thead className="bg-slate-100 text-slate-600 sticky top-0 shadow-sm">
                                  <tr>
                                    <th className="px-3 py-2 font-medium">Rang</th>
                                    <th className="px-3 py-2 font-medium">Bel.</th>
                                    <th className="px-3 py-2 font-medium">Verstärkung</th>
                                    <th className="px-3 py-2 font-medium">Laser</th>
                                    <th className="px-3 py-2 font-medium">Leistung</th>
                                    <th className="px-3 py-2 font-medium">Rohdaten-Zuverlässigkeit</th>
                                    <th className="px-3 py-2 font-medium">Fehler</th>
                                    <th title="Leer P95: In 95 % der Rohbilder lag die Abdeckung höchstens bei diesem Wert. Kleiner ist besser. Gefüllt P05: In 95 % der Rohbilder lag sie mindestens bei diesem Wert. Größer ist besser." className="px-3 py-2 font-medium">Messabdeckung</th>
                                    <th className="px-3 py-2 font-medium">Flickern</th>
                                    <th className="px-3 py-2 font-medium">Vorgeschlagener Schwellwert</th>
                                    <th className="px-3 py-2 font-medium text-right">Aktion</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-200">
                                  {status.autocalibrate_results.map((res: any, idx: number) => (
                                    <tr key={idx} className="hover:bg-white transition-colors">
                                      <td className="px-3 py-2 font-mono text-slate-500">#{idx + 1}</td>
                                      <td className="px-3 py-2">{res.exposure}</td>
                                      <td className="px-3 py-2">{res.gain}</td>
                                      <td className="px-3 py-2">{res.laser}</td>
                                      <td className={`px-3 py-2 font-bold ${res.performance_score >= 95 ? 'text-emerald-600' : res.performance_score >= 80 ? 'text-amber-600' : 'text-red-600'}`} title="100 bedeutet: jedes Rohbild war korrekt und es gab kein Flickern. Niedrigere Werte bestrafen unzuverlässige Löcher, falsche Bilder, Fehler des schlechtesten Lochs und Zustandswechsel.">{res.performance_score?.toFixed(1) ?? '—'}<span className="text-slate-400 font-normal"> / 100</span></td>
                                      <td className="px-3 py-2 font-semibold text-emerald-600">
                                        {res.empty_reliable_holes !== undefined ? `${res.empty_reliable_holes}/42 offen · ${res.filled_reliable_holes}/42 gefüllt` : `${res.score}/42 roh-offen`}
                                      </td>
                                      <td className="px-3 py-2">{res.raw_errors ?? '—'}<span className="text-slate-400 ml-1">schlechtestes {Math.round((res.worst_error_rate || 0) * 100)}%</span></td>
                                      <td className="px-3 py-2">{res.filled_p05_coverage !== undefined ? `leer ≤${Math.round(res.empty_p95_coverage * 100)}% · gefüllt ≥${Math.round(res.filled_p05_coverage * 100)}%` : `leer ≤${Math.round((res.empty_p95_coverage || 0) * 100)}%`}</td>
                                      <td className="px-3 py-2">{res.flicker_transitions ?? '—'} Wechsel</td>
                                      <td className="px-3 py-2 font-mono">{res.suggested_occupancy_threshold ?? '—'}</td>
                                      <td className="px-3 py-2 text-right">
                                        <div className="flex justify-end gap-2">
                                          <Button title="Übernimmt diese Belichtung, diesen Gain und diese Laserleistung vorübergehend für die Kamera." size="sm" variant="outline" className="h-7 text-xs border-[#b1ca21] text-[#8a9e19] hover:bg-[#b1ca21] hover:text-white" onClick={() => action('autocalibrate/use_result', { index: idx })}>Kamera verwenden</Button>
                                          {res.suggested_occupancy_threshold !== null && res.suggested_occupancy_threshold !== undefined && (
                                            <Button size="sm" variant="outline" className="h-7 text-xs" title="Schreibt diesen Wert in calibration.json. Das Spiel lädt ihn beim Start." onClick={() => action('autocalibrate/use_result', { index: idx, apply_suggested_threshold: true, save_threshold_for_game: true })}>Schwellwert fürs Spiel übernehmen</Button>
                                          )}
                                        </div>
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
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Schnelles Scannen des Felds …
                            </Button>
                            <Button title="Bricht die laufende Kalibrierung ab und stellt die vorherigen Tiefensensorwerte wieder her." variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Abbrechen</Button>
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
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin text-slate-500" /> Schritt 1: Leeres Feld wird gescannt …
                            </Button>
                            <Button title="Bricht die laufende Kalibrierung ab und stellt die vorherigen Tiefensensorwerte wieder her." variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Abbrechen</Button>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-[#b1ca21] h-2 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                          </div>
                        </div>
                      )}

                      {status.autocalibrate_state === 2 && (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-4">
                            <Button title="Startet die zweite Messung, nachdem alle 42 Löcher mit Steinen gefüllt wurden." size="lg" onClick={() => action('autocalibrate_step2')} className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white px-6 animate-pulse">
                              <RefreshCw className="w-4 h-4 mr-2" /> Gefülltes Feld scannen (Schritt 2)
                            </Button>
                            <Button title="Bricht die laufende Kalibrierung ab und stellt die vorherigen Tiefensensorwerte wieder her." variant="outline" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Abbrechen</Button>
                          </div>
                        </div>
                      )}

                      {status.autocalibrate_state === 3 && (
                        <div className="flex flex-col gap-3">
                          <div className="flex items-center gap-2">
                            <Button size="lg" disabled className="bg-slate-300 text-slate-500 px-6 flex-1">
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" /> Schritt 2: Gefülltes Feld wird gescannt …
                            </Button>
                            <Button title="Bricht die laufende Kalibrierung ab und stellt die vorherigen Tiefensensorwerte wieder her." variant="outline" size="lg" onClick={cancelAutocalibrate} className="text-red-500 border-red-200">Abbrechen</Button>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-[#b1ca21] h-2 rounded-full transition-all duration-300" style={{ width: `${(status.autocalibrate_progress || 0) * 100}%` }}></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>
              )}

              {realsenseSubTab === 'manual' && (
                <div className="space-y-6">
                  <Card className="bg-white border-slate-200 shadow-sm rounded-tl-none">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                      <CardTitle className="text-lg text-slate-800">Manuelle Tiefensensor-Einstellungen</CardTitle>
                      <CardDescription className="text-slate-500">Nur Tiefenbelichtung, Gain, Projektorleistung und Hardware-Voreinstellung.</CardDescription>
                    </CardHeader>
                    <CardContent className="flex flex-col gap-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <SliderWithInput label="Belichtung" description="Belichtungszeit der Tiefenkamera. Niedrigere Werte verringern Bewegungsunschärfe, machen das Bild aber dunkler." value={pendingOverrides.exposure !== undefined ? pendingOverrides.exposure : status.exposure} min={1} max={10000} step={50} disabled={manualControlsLocked} onChange={(v) => setPendingOverrides({...pendingOverrides, exposure: v})} />
                        <SliderWithInput label="Verstärkung" description="Signalverstärkung des Sensors. Verstärkt das Signal, kann aber Rauschen erhöhen." value={pendingOverrides.gain !== undefined ? pendingOverrides.gain : status.gain} min={16} max={248} disabled={manualControlsLocked} onChange={(v) => setPendingOverrides({...pendingOverrides, gain: v})} />
                        <SliderWithInput label="Laserleistung" description="Intensität des Infrarot-Projektors für die Tiefenschätzung." value={pendingOverrides.laser_power !== undefined ? pendingOverrides.laser_power : status.laser_power} min={0} max={360} disabled={manualControlsLocked} onChange={(v) => setPendingOverrides({...pendingOverrides, laser_power: v})} />
                        <SliderWithInput label="Hardware-Voreinstellung" description="Hardware-Optimierungsvoreinstellung (3 = hohe Genauigkeit)." value={pendingOverrides.visual_preset !== undefined ? pendingOverrides.visual_preset : status.visual_preset} min={0} max={5} disabled={manualControlsLocked} onChange={(v) => setPendingOverrides({...pendingOverrides, visual_preset: v})} />
                      </div>
                      <div className="flex justify-end mt-2">
                        <Button 
                          title="Überträgt die geänderten manuellen Werte an die Tiefenkamera. Sie werden erst mit dem Speicherknopf oben dauerhaft gesichert."
                          className="bg-[#b1ca21] hover:bg-[#a0b51e] text-white disabled:opacity-50"
                          disabled={manualControlsLocked || Object.keys(pendingOverrides).length === 0}
                          onClick={() => {
                            if (Object.keys(pendingOverrides).length === 0) return;
                            fetch('/api/update_realsense', {
                              method: 'POST',
                              headers: { 'Content-Type': 'application/json' },
                              body: JSON.stringify(pendingOverrides)
                            }).then(() => setPendingOverrides({}))
                          }}
                        >
                          Manuelle Einstellungen anwenden
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {realsenseSubTab === 'filtering' && (
                <div className="space-y-6">
                  <Card className="bg-white border-slate-200 shadow-sm rounded-tl-none flex flex-col">
                    <CardHeader className="pb-4 border-b border-slate-100 mb-4">
                      <CardTitle className="text-lg text-slate-800">Tiefenfilter</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <SliderWithInput label="Minimale Tiefe (mm)" description="Alle Pixel näher als diese Entfernung ignorieren." value={status.min_depth} max={5000} step={10} disabled={manualControlsLocked} onChange={(v) => updateRealSense('min_depth', v)} />
                      <SliderWithInput label="Maximale Tiefe (mm)" description="Alle Pixel weiter entfernt als diese Entfernung ignorieren." value={status.max_depth} max={5000} step={10} disabled={manualControlsLocked} onChange={(v) => updateRealSense('max_depth', v)} />
                      <div className="mt-6 p-5 bg-white rounded-xl border border-slate-200 flex items-center justify-between shadow-sm">
                        <Label className="text-slate-700 font-medium tracking-wide text-sm cursor-pointer" htmlFor="emitter-toggle">
                          Infrarot-Emitter aktiviert
                        </Label>
                        <input title="Schaltet den Infrarot-Projektor der Tiefenkamera ein oder aus." id="emitter-toggle" type="checkbox" disabled={manualControlsLocked} checked={status.emitter === 1} onChange={(e) => updateRealSense('emitter', e.target.checked ? 1 : 0)} className="w-5 h-5 rounded" />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
                </div>
              </div>

            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
