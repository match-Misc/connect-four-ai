import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, Link } from 'react-router-dom';
import { cn } from './utils';
import { ConnectFourGrid } from './components/ConnectFourGrid';
import { Activity, Bot, User, Settings2, Bug, Gamepad2 } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000/api`;

function GameBoard({ showDebug }: { showDebug: boolean }) {
  const [board, setBoard] = useState<number[][]>(Array(6).fill(Array(7).fill(0)));
  const [turn, setTurn] = useState<string>('human');
  const [robotState, setRobotState] = useState<string>('idle');
  const [simulationMode, setSimulationMode] = useState<boolean>(true);
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [winner, setWinner] = useState<number | null>(null);
  const [matchState, setMatchState] = useState<string>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [invalidStones, setInvalidStones] = useState<number[][]>([]);
  const [debounceTime, setDebounceTime] = useState<number>(1.0);
  const [aiEnabled, setAiEnabled] = useState<boolean>(true);
  const [robotTargetCol, setRobotTargetCol] = useState<number | null>(null);

  const fetchBoardState = async () => {
    try {
      const res = await fetch(`${API_BASE}/board-state`);
      const data = await res.json();
      setBoard(data.board);
      setTurn(data.turn);
      setRobotState(data.robot_state);
      setSimulationMode(data.simulation_mode);
      setGameOver(data.game_over || false);
      setWinner(data.winner || null);
      setMatchState(data.match_state || 'idle');
      setErrorMsg(data.error_msg || null);
      setInvalidStones(data.invalid_stones || []);
      if (data.debounce_time !== undefined) {
        setDebounceTime(data.debounce_time);
      }
      if (data.ai_enabled !== undefined) {
        setAiEnabled(data.ai_enabled);
      }
      setRobotTargetCol(data.robot_target_col ?? null);
    } catch (e) {
      console.error('Failed to fetch board state:', e);
    }
  };

  useEffect(() => {
    const interval = setInterval(fetchBoardState, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (aiEnabled && turn === 'robot' && robotState === 'idle') {
      triggerRobotMove();
    }
  }, [aiEnabled, turn, robotState]);

  const triggerRobotMove = async () => {
    try {
      await fetch(`${API_BASE}/robot-move`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to trigger robot move:', e);
    }
  };

  const handleColumnClick = async (col: number) => {
    if (turn !== 'human' || gameOver) return;
    try {
      const res = await fetch(`${API_BASE}/player-move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ column: col, player: 1 }),
      });
      const data = await res.json();
      setBoard(data.board);
      setTurn('robot');
    } catch (e) {
      console.error('Failed to submit move:', e);
    }
  };

  const updateConfig = async () => {
    try {
      await fetch(`${API_BASE}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulate: simulationMode, difficulty, debounce_time: debounceTime, ai_enabled: aiEnabled }),
      });
    } catch (e) {
      console.error('Failed to update config:', e);
    }
  };

  useEffect(() => {
    updateConfig();
  }, [simulationMode, difficulty, debounceTime, aiEnabled]);

  return (
    <div className={cn("bg-gray-50 flex flex-col items-center py-4 font-sans w-full", showDebug ? "min-h-screen" : "h-screen overflow-hidden")}>
      <div className="w-full px-4 xl:px-8 flex flex-col gap-4 h-full">
        
        {/* Header */}
        <header className="flex flex-col 2xl:flex-row justify-between items-center bg-white p-4 lg:p-6 rounded-2xl shadow-sm border border-gray-100 gap-4 shrink-0">
          <div className="flex items-center gap-4">
            <img src="/match_NUR_Logo_10pt.svg" alt="Match Logo" className="h-10" onError={(e) => (e.currentTarget.style.display = 'none')} />
            <div className="flex flex-col">
              <h1 className="text-xl lg:text-2xl font-bold text-gray-800 tracking-tight whitespace-nowrap">Connect Four AI</h1>
              <div className="flex gap-4 mt-1">
                <Link to="/game" className={cn("text-xs font-semibold flex items-center gap-1 transition-colors", !showDebug ? "text-brand-green" : "text-gray-400 hover:text-gray-600")}>
                  <Gamepad2 size={14} /> Play
                </Link>
                <Link to="/debugging" className={cn("text-xs font-semibold flex items-center gap-1 transition-colors", showDebug ? "text-purple-600" : "text-gray-400 hover:text-gray-600")}>
                  <Bug size={14} /> Debug
                </Link>
              </div>
            </div>
          </div>
          
          <div className="flex flex-wrap items-center justify-center gap-4 lg:gap-6 w-full 2xl:w-auto">
            {/* Difficulty Selector */}
            <div className="flex bg-gray-100 p-1 rounded-lg">
              {['easy', 'medium', 'hard', 'impossible'].map(level => (
                <button
                  key={level}
                  onClick={() => setDifficulty(level)}
                  className={cn(
                    "px-3 py-1.5 lg:px-4 lg:py-2 rounded-md text-sm font-bold capitalize transition-all", 
                    difficulty === level ? "bg-white shadow-sm text-brand-green" : "text-gray-500 hover:text-gray-700"
                  )}
                >
                  {level}
                </button>
              ))}
            </div>

            {/* Game Controls */}
            <div className="flex gap-2">
              <button 
                onClick={async () => {
                  await fetch(`${API_BASE}/start`, { method: 'POST' });
                }}
                className="bg-brand-green hover:bg-[#7a8a14] text-white px-4 py-2 lg:px-6 lg:py-2 rounded-lg text-sm font-bold transition-colors shadow-sm"
              >
                Start Game
              </button>
              <button 
                onClick={async () => {
                  await fetch(`${API_BASE}/reset`, { method: 'POST' });
                }}
                className="bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 lg:px-6 lg:py-2 rounded-lg text-sm font-bold transition-colors shadow-sm"
              >
                Reset
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 lg:gap-4">
            <span className={cn("px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider whitespace-nowrap", 
              matchState === 'idle' ? 'bg-gray-100 text-gray-500 border border-gray-200' :
              matchState === 'in_game' ? 'bg-blue-100 text-blue-700 border border-blue-200 shadow-sm' :
              'bg-yellow-100 text-yellow-800 border border-yellow-300 shadow-sm'
            )}>
              {matchState === 'idle' ? 'Waiting to Start' : matchState === 'in_game' ? 'In Game' : 'Finished'}
            </span>
            <div className={cn(
              "flex items-center gap-2 px-3 py-2 lg:px-4 lg:py-2 rounded-full font-bold text-sm transition-colors whitespace-nowrap",
              turn === 'human' ? "bg-brand-green/20 text-green-900 border border-brand-green" : "bg-gray-100 text-gray-500"
            )}>
              <User size={16} /> Human Turn
            </div>
            <div className={cn(
              "flex items-center gap-2 px-3 py-2 lg:px-4 lg:py-2 rounded-full font-bold text-sm transition-colors whitespace-nowrap",
              turn === 'robot' ? "bg-gray-800 text-white shadow-lg" : "bg-gray-100 text-gray-500"
            )}>
              <Bot size={16} /> Robot Turn
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-col gap-4 items-center w-full flex-1 min-h-0">
          {errorMsg && (
            <div className="w-full bg-red-100 p-4 rounded-xl shadow-sm border-2 border-red-500 flex items-center justify-center animate-pulse shrink-0">
              <span className="text-red-700 font-bold text-lg flex items-center gap-2">
                ⚠ {errorMsg}
              </span>
            </div>
          )}
          
          {gameOver && !errorMsg && (
            <div className="w-full bg-white p-4 rounded-2xl shadow-xl border-4 border-yellow-400 flex flex-col items-center justify-center animate-bounce-short shrink-0">
              <h2 className="text-2xl lg:text-3xl font-black text-gray-800 uppercase tracking-widest mb-1 lg:mb-2">Game Over!</h2>
              <div className="text-lg lg:text-xl font-bold">
                {winner === 1 ? (
                  <span className="text-brand-green flex items-center gap-2">🎉 Human Wins! 🎉</span>
                ) : winner === 2 ? (
                  <span className="text-red-500 flex items-center gap-2">🤖 Robot Wins! 🤖</span>
                ) : (
                  <span className="text-gray-500 flex items-center gap-2">🤝 It's a Draw! 🤝</span>
                )}
              </div>
            </div>
          )}
          
          <div className="w-full flex-1 flex justify-center items-center min-h-0">
            <div className="h-full max-w-full aspect-[7/6] transition-all duration-300 ease-in-out">
              <ConnectFourGrid 
                board={board} 
                onColumnClick={handleColumnClick}
                disabled={turn !== 'human' || gameOver}
                invalidStones={invalidStones}
                robotTargetCol={simulationMode && aiEnabled ? robotTargetCol : null}
              />
            </div>
          </div>

          {/* Debug View Content */}
          {showDebug && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full mt-8">
              
              <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col gap-4">
                <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                  <Activity size={20} className="text-purple-500" /> Vision Feed
                </h2>
                <div className="text-sm text-gray-600 flex gap-4">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-white inline-block bg-gray-200"></span> Empty</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-green-500 inline-block bg-transparent"></span> P1 Detected</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-red-500 inline-block bg-transparent"></span> P2 Detected</span>
                </div>
                <div className="rounded-lg overflow-hidden border border-gray-200 bg-gray-100 aspect-[4/3] flex items-center justify-center relative w-full">
                  <img 
                    src={`${API_BASE.replace('/api', '')}/api/video-feed`} 
                    alt="RealSense Stream" 
                    className="absolute top-0 left-0 w-full h-full object-cover"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                      e.currentTarget.parentElement!.innerHTML = '<span class="text-xs text-gray-500">Camera Feed Offline</span>';
                    }}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-6">
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                    <Activity size={20} className="text-blue-500" /> Robot Status
                  </h2>
                  
                  <div className="flex flex-col gap-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 font-bold">State</span>
                      <span className={cn(
                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                        robotState === 'idle' ? "bg-gray-100 text-gray-600" :
                        robotState === 'analyzing' ? "bg-blue-100 text-blue-700 animate-pulse" :
                        robotState === 'thinking' ? "bg-purple-100 text-purple-700 animate-pulse" :
                        "bg-orange-100 text-orange-700 animate-bounce"
                      )}>
                        {robotState}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                    <Settings2 size={20} className="text-gray-500" /> Settings
                  </h2>
                  
                  <label className="flex items-center justify-between cursor-pointer group">
                    <span className="text-sm font-bold text-gray-700 group-hover:text-gray-900 transition-colors">AI Enabled</span>
                    <div className="relative">
                      <input 
                        type="checkbox" 
                        className="sr-only" 
                        checked={aiEnabled}
                        onChange={(e) => setAiEnabled(e.target.checked)}
                      />
                      <div className={cn(
                        "block w-10 h-6 rounded-full transition-colors",
                        aiEnabled ? "bg-brand-green" : "bg-gray-300"
                      )}></div>
                      <div className={cn(
                        "absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform",
                        aiEnabled ? "transform translate-x-4" : ""
                      )}></div>
                    </div>
                  </label>
                  
                  <label className="flex items-center justify-between cursor-pointer group">
                    <span className="text-sm font-bold text-gray-700 group-hover:text-gray-900 transition-colors">Simulation Mode</span>
                    <div className="relative">
                      <input 
                        type="checkbox" 
                        className="sr-only" 
                        checked={simulationMode}
                        onChange={(e) => setSimulationMode(e.target.checked)}
                      />
                      <div className={cn(
                        "block w-10 h-6 rounded-full transition-colors",
                        simulationMode ? "bg-brand-green" : "bg-gray-300"
                      )}></div>
                      <div className={cn(
                        "absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform",
                        simulationMode ? "transform translate-x-4" : ""
                      )}></div>
                    </div>
                  </label>
                  
                  <div className="flex flex-col gap-2 pt-2 border-t border-gray-100">
                    <label className="text-sm font-bold text-gray-700 flex justify-between">
                      <span>CV Debounce Delay</span>
                      <span className="text-brand-green">{debounceTime.toFixed(1)}s</span>
                    </label>
                    <input 
                      type="range" 
                      min="0.1" max="3.0" step="0.1" 
                      value={debounceTime}
                      onChange={(e) => setDebounceTime(parseFloat(e.target.value))}
                      className="w-full accent-brand-green"
                    />
                    <p className="text-xs text-gray-500">Delay before accepting a newly detected token to prevent mid-air bugs.</p>
                  </div>
                  
                  {simulationMode && aiEnabled && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
                      <span className="font-bold">⚠</span>
                      <p>AI Advisory active. The AI will advise you via the UI, but you must physically place its black token.</p>
                    </div>
                  )}
                  {simulationMode && !aiEnabled && (
                    <div className="bg-blue-50 border border-blue-200 text-blue-800 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
                      <span className="font-bold">ℹ</span>
                      <p>2-Player Manual Mode. Play entirely physically without AI interference.</p>
                    </div>
                  )}
                </div>
              </div>
              
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/game" replace />} />
        <Route path="/game" element={<GameBoard showDebug={false} />} />
        <Route path="/debugging" element={<GameBoard showDebug={true} />} />
      </Routes>
    </Router>
  );
}

export default App;
