import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, Link } from 'react-router-dom';
import { cn } from './utils';
import { ConnectFourGrid } from './components/ConnectFourGrid';
import { Fireworks } from './components/Fireworks';
import { RobotWins } from './components/RobotWins';
import { ThemeToggle } from './components/ThemeToggle';
import { Activity, Bot, User, Settings2, Bug, Gamepad2, Sparkles, Nfc } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000/api`;

function RobotArmIcon({ size = 24, className = "" }: { size?: number, className?: string }) {
  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      className={className}
    >
      <path d="M6 22h12" />
      <path d="M9 22v-3a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v3" />
      <circle cx="12" cy="14" r="2" />
      <path d="M12 12L9.5 7" />
      <circle cx="9" cy="6" r="2" />
      <path d="M10 5L15 4" />
      <circle cx="16" cy="4" r="1.5" />
      <path d="M17.5 4h2" />
      <path d="M19.5 3v2" />
    </svg>
  );
}

function GameBoard({ showDebug }: { showDebug: boolean }) {
  const [board, setBoard] = useState<number[][]>(Array(6).fill(Array(7).fill(0)));
  const [turn, setTurn] = useState<string>('human');
  const [robotState, setRobotState] = useState<string>('idle');
  const [simulationMode, setSimulationMode] = useState<boolean>(false);
  const [difficulty, setDifficulty] = useState<string>('medium');
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [winner, setWinner] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [invalidStones, setInvalidStones] = useState<number[][]>([]);
  const [debounceTime, setDebounceTime] = useState<number>(0.5);
  const [aiEnabled, setAiEnabled] = useState<boolean>(true);
  const [robotTargetCol, setRobotTargetCol] = useState<number | null>(null);
  const [tcpConnected, setTcpConnected] = useState<boolean>(false);
  const [nfcConnected, setNfcConnected] = useState<boolean>(false);
  const [celebrating, setCelebrating] = useState<boolean>(false);
  const [consoling, setConsoling] = useState<boolean>(false);

  const [nfcData, setNfcData] = useState<string | null>(null);
  const [nfcInvalidScanTime, setNfcInvalidScanTime] = useState<number>(0);
  const [nfcTimeout, setNfcTimeout] = useState<number>(15.0);
  const [blinkNfc, setBlinkNfc] = useState<boolean>(false);
  const [nfcOverwritten, setNfcOverwritten] = useState<boolean>(false);
  // Until the first board-state response lands, our config values are placeholders
  // rather than the persisted ones — posting them would overwrite the saved session.
  const hydrated = useRef(false);
  // The board state is polled, so the result arrives over and over — only the
  // first one of a game should set off an end-of-game animation.
  const gameOverAnimated = useRef(false);

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
      setErrorMsg(data.error_msg || null);
      setInvalidStones(data.invalid_stones || []);
      if (data.debounce_time !== undefined) {
        setDebounceTime(data.debounce_time);
      }
      if (data.ai_enabled !== undefined) {
        setAiEnabled(data.ai_enabled);
      }
      if (data.difficulty !== undefined) {
        setDifficulty(data.difficulty);
      }
      if (data.tcp_connected !== undefined) {
        setTcpConnected(data.tcp_connected);
      }
      if (data.nfc_connected !== undefined) {
        setNfcConnected(data.nfc_connected);
      }
      if (data.nfc_data !== undefined) {
        setNfcData(prev => {
          if (prev !== null && data.nfc_data !== null && prev !== data.nfc_data) {
            setNfcOverwritten(true);
            setTimeout(() => setNfcOverwritten(false), 2000);
          }
          return data.nfc_data;
        });
      }
      if (data.nfc_invalid_scan_time !== undefined) {
        if (data.nfc_invalid_scan_time !== nfcInvalidScanTime && data.nfc_invalid_scan_time > 0) {
          setNfcInvalidScanTime(data.nfc_invalid_scan_time);
          setBlinkNfc(true);
          setTimeout(() => setBlinkNfc(false), 3000);
        }
      }
      if (data.nfc_timeout !== undefined) {
        setNfcTimeout(data.nfc_timeout);
      }
      setRobotTargetCol(data.robot_target_col ?? null);
      hydrated.current = true;
    } catch (e) {
      console.error('Failed to fetch board state:', e);
    }
  };

  useEffect(() => {
    fetchBoardState();
    // The backend advances detection on its own thread now, so this only sets
    // how fast the GUI catches up to it. Each poll re-renders the whole board,
    // so we back off while the celebration runs — the game is over anyway, and
    // the animation gets the main thread to itself.
    const interval = setInterval(fetchBoardState, celebrating || consoling ? 600 : 150);
    return () => clearInterval(interval);
  }, [celebrating, consoling]);

  useEffect(() => {
    if (aiEnabled && turn === 'robot' && robotState === 'idle') {
      triggerRobotMove();
    }
  }, [aiEnabled, turn, robotState]);

  useEffect(() => {
    if (!gameOver) {
      // Re-arm for the next game.
      gameOverAnimated.current = false;
      return;
    }
    if (gameOverAnimated.current) return;
    if (winner === 1) {
      gameOverAnimated.current = true;
      setCelebrating(true);
    } else if (winner === 2) {
      gameOverAnimated.current = true;
      setConsoling(true);
    }
  }, [gameOver, winner]);

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
        body: JSON.stringify({ simulate: simulationMode, difficulty, debounce_time: debounceTime, ai_enabled: aiEnabled, nfc_timeout: nfcTimeout }),
      });
    } catch (e) {
      console.error('Failed to update config:', e);
    }
  };

  useEffect(() => {
    if (!hydrated.current) return;
    updateConfig();
  }, [simulationMode, difficulty, debounceTime, aiEnabled, nfcTimeout]);

  return (
    <div className={cn("bg-gray-50 dark:bg-gray-950 flex flex-col items-center py-4 font-sans w-full", showDebug ? "min-h-screen" : "h-screen overflow-hidden")}>
      <Fireworks active={celebrating} onDone={() => setCelebrating(false)} />
      <RobotWins active={consoling} onDone={() => setConsoling(false)} />
      {celebrating && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center pointer-events-none">
          <h2 className="animate-win-text text-center font-black uppercase tracking-tight text-brand-green text-5xl sm:text-7xl lg:text-8xl drop-shadow-[0_4px_20px_rgba(0,0,0,0.6)]">
            Du hast gewonnen!
          </h2>
        </div>
      )}
      <div className="w-full px-4 xl:px-8 flex flex-col gap-4 h-full">

        {/* Header */}
        <header className="relative flex flex-col 2xl:flex-row portrait:2xl:flex-col justify-between items-center bg-white dark:bg-gray-900 p-4 lg:p-6 portrait:p-4 pr-14 lg:pr-16 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 gap-4 portrait:gap-3 shrink-0">
          <ThemeToggle className="absolute top-3 right-3" />
          <div className="flex items-center gap-5">
            <img src="/match_NUR_Logo_10pt.svg" alt="Match Logo" className="h-14 lg:h-20" onError={(e) => (e.currentTarget.style.display = 'none')} />
            <div className="flex flex-col">
              <h1 className="text-2xl lg:text-4xl font-bold text-gray-800 dark:text-gray-100 tracking-tight whitespace-nowrap">Connect Four AI</h1>
              <div className="flex gap-5 mt-1.5 items-center">
                <Link to="/game" className={cn("text-sm lg:text-base font-semibold flex items-center gap-1.5 transition-colors", !showDebug ? "text-brand-green" : "text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300")}>
                  <Gamepad2 size={18} /> Play
                </Link>
                <Link to={showDebug ? "/game" : "/debugging"} title={showDebug ? "Close debug view" : "Open debug view"} className={cn("text-sm lg:text-base font-semibold flex items-center gap-1.5 transition-colors", showDebug ? "text-purple-600 dark:text-purple-400" : "text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300")}>
                  <Bug size={18} /> Debug
                </Link>
                <div className="h-4 w-px bg-gray-300 dark:bg-gray-700 mx-1" />
                <div className="flex items-center gap-1.5" title={tcpConnected ? "Robot Connected" : "Robot Disconnected"}>
                  <RobotArmIcon size={16} className="text-gray-500 dark:text-gray-400" />
                  <span 
                    className={cn(
                      "w-2 h-2 rounded-full",
                      tcpConnected ? "bg-green-500" : "bg-red-500"
                    )}
                  />
                </div>
                <div className="flex items-center gap-1.5 ml-2" title={nfcConnected ? "NFC Connected" : "NFC Disconnected"}>
                  <Nfc size={16} className="text-gray-500 dark:text-gray-400" />
                  <span 
                    className={cn(
                      "w-2 h-2 rounded-full",
                      nfcConnected ? "bg-green-500" : "bg-red-500"
                    )}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 lg:gap-6 w-full 2xl:w-auto portrait:2xl:w-full">
            {/* Difficulty Selector */}
            <div className="flex bg-gray-100 dark:bg-gray-800 p-1.5 rounded-xl">
              {['easy', 'medium', 'hard', 'impossible'].map(level => (
                <button
                  key={level}
                  onClick={() => setDifficulty(level)}
                  className={cn(
                    "px-5 py-2.5 lg:px-7 lg:py-3.5 rounded-lg text-lg lg:text-xl font-bold capitalize transition-all",
                    difficulty === level ? "bg-white dark:bg-gray-700 shadow-sm text-brand-green" : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
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
                  await fetch(`${API_BASE}/reset`, { method: 'POST' });
                }}
                className="bg-red-100 hover:bg-red-200 text-red-700 dark:bg-red-950 dark:hover:bg-red-900 dark:text-red-300 px-6 py-3 lg:px-8 lg:py-4 rounded-xl text-lg lg:text-xl font-bold transition-colors shadow-sm"
              >
                Reset
              </button>

              {/* TEMPORARY: previews the end-of-game animations without playing a game. Remove once they are signed off. */}
              <button
                onClick={() => {
                  setCelebrating(false);
                  // Remount on the next frame so a second click restarts the show.
                  requestAnimationFrame(() => setCelebrating(true));
                }}
                title="Temporary: preview the win animation"
                className="bg-amber-100 hover:bg-amber-200 text-amber-700 dark:bg-amber-950 dark:hover:bg-amber-900 dark:text-amber-300 px-6 py-3 lg:px-8 lg:py-4 rounded-xl text-lg lg:text-xl font-bold transition-colors shadow-sm flex items-center gap-2"
              >
                <Sparkles className="w-5 h-5 lg:w-6 lg:h-6" /> Win
              </button>
              <button
                onClick={() => {
                  setConsoling(false);
                  requestAnimationFrame(() => setConsoling(true));
                }}
                title="Temporary: preview the loss animation"
                className="bg-sky-100 hover:bg-sky-200 text-sky-700 dark:bg-sky-950 dark:hover:bg-sky-900 dark:text-sky-300 px-6 py-3 lg:px-8 lg:py-4 rounded-xl text-lg lg:text-xl font-bold transition-colors shadow-sm flex items-center gap-2"
              >
                <Bot className="w-5 h-5 lg:w-6 lg:h-6" /> Lose
              </button>
            </div>
          </div>

          <div className="flex items-center w-full mt-4 lg:mt-6 2xl:mt-0 portrait:2xl:mt-4">
            {/* Left spacer / NFC tag */}
            <div className="flex-1 flex justify-end pr-8 lg:pr-12">
              <div className={cn(
                "flex items-center gap-3 font-bold text-xl lg:text-2xl uppercase tracking-wide whitespace-nowrap transition-all duration-300",
                nfcOverwritten
                  ? "text-orange-500"
                  : nfcData
                  ? "text-brand-green"
                  : "text-gray-400 dark:text-gray-600",
                blinkNfc ? "animate-pulse text-red-500" : ""
              )}>
                <Nfc className="w-8 h-8 lg:w-10 lg:h-10" />
                {nfcOverwritten ? "Überschrieben" : nfcData ? "NFC Registriert" : "NFC Tag"}
              </div>
            </div>

            {/* Center aligned items */}
            <div className="flex items-center gap-8 lg:gap-16">
              <div className={cn(
                "flex items-center gap-4 px-8 py-5 lg:px-10 lg:py-6 rounded-2xl font-black text-3xl lg:text-4xl uppercase tracking-wide whitespace-nowrap transition-all duration-300",
                turn === 'human'
                  ? "bg-brand-green/20 text-green-900 dark:text-brand-green border-4 border-brand-green scale-105 shadow-lg"
                  : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600 border-4 border-transparent opacity-60"
              )}>
                <User className="w-9 h-9 lg:w-12 lg:h-12" /> Human
              </div>
              <div className={cn(
                "flex items-center gap-4 px-8 py-5 lg:px-10 lg:py-6 rounded-2xl font-black text-3xl lg:text-4xl uppercase tracking-wide whitespace-nowrap transition-all duration-300",
                turn === 'robot'
                  ? "bg-gray-800 text-white dark:bg-gray-100 dark:text-gray-900 border-4 border-gray-800 dark:border-gray-100 scale-105 shadow-lg"
                  : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600 border-4 border-transparent opacity-60"
              )}>
                <Bot className="w-9 h-9 lg:w-12 lg:h-12" /> Robot
              </div>
            </div>

            {/* Right spacer for perfect center balance */}
            <div className="flex-1"></div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex flex-col gap-4 items-center w-full flex-1 min-h-0">
          {errorMsg && (
            <div className="w-full bg-red-100 dark:bg-red-950 p-4 rounded-xl shadow-sm border-2 border-red-500 flex items-center justify-center animate-pulse shrink-0">
              <span className="text-red-700 dark:text-red-300 font-bold text-lg flex items-center gap-2">
                ⚠ {errorMsg}
              </span>
            </div>
          )}
          
          {gameOver && !errorMsg && (
            <div className="w-full bg-white dark:bg-gray-900 p-4 rounded-2xl shadow-xl border-4 border-yellow-400 flex flex-col items-center justify-center animate-bounce-short shrink-0">
              <h2 className="text-2xl lg:text-3xl font-black text-gray-800 dark:text-gray-100 uppercase tracking-widest mb-1 lg:mb-2">Game Over!</h2>
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
            <div className="aspect-[7/6] h-full w-auto max-w-full portrait:h-auto portrait:w-full portrait:max-h-full transition-all duration-300 ease-in-out">
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
            <div className="grid grid-cols-1 lg:grid-cols-3 portrait:grid-cols-1 gap-6 w-full mt-8">
              
              <div className="lg:col-span-2 portrait:col-span-1 bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                  <Activity size={20} className="text-purple-500" /> Vision Feed
                </h2>
                <div className="text-sm text-gray-600 dark:text-gray-400 flex gap-4">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-white dark:border-gray-600 inline-block bg-gray-200 dark:bg-gray-700"></span> Empty</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-green-500 inline-block bg-transparent"></span> P1 Detected</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-red-500 inline-block bg-transparent"></span> P2 Detected</span>
                </div>
                <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 aspect-[4/3] flex items-center justify-center relative w-full">
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
                <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <Activity size={20} className="text-blue-500" /> Robot Status
                  </h2>

                  <div className="flex flex-col gap-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">State</span>
                      <span className={cn(
                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                        robotState === 'idle' ? "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300" :
                        robotState === 'analyzing' ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 animate-pulse" :
                        robotState === 'thinking' ? "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300 animate-pulse" :
                        "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300 animate-bounce"
                      )}>
                        {robotState}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <Settings2 size={20} className="text-gray-500" /> Settings
                  </h2>

                  <label className="flex items-center justify-between cursor-pointer group">
                    <span className="text-sm font-bold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">AI Enabled</span>
                    <div className="relative">
                      <input 
                        type="checkbox" 
                        className="sr-only" 
                        checked={aiEnabled}
                        onChange={(e) => setAiEnabled(e.target.checked)}
                      />
                      <div className={cn(
                        "block w-10 h-6 rounded-full transition-colors",
                        aiEnabled ? "bg-brand-green" : "bg-gray-300 dark:bg-gray-700"
                      )}></div>
                      <div className={cn(
                        "absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform",
                        aiEnabled ? "transform translate-x-4" : ""
                      )}></div>
                    </div>
                  </label>
                  
                  <label className="flex items-center justify-between cursor-pointer group">
                    <span className="text-sm font-bold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">Simulation Mode</span>
                    <div className="relative">
                      <input 
                        type="checkbox" 
                        className="sr-only" 
                        checked={simulationMode}
                        onChange={(e) => setSimulationMode(e.target.checked)}
                      />
                      <div className={cn(
                        "block w-10 h-6 rounded-full transition-colors",
                        simulationMode ? "bg-brand-green" : "bg-gray-300 dark:bg-gray-700"
                      )}></div>
                      <div className={cn(
                        "absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform",
                        simulationMode ? "transform translate-x-4" : ""
                      )}></div>
                    </div>
                  </label>
                  
                  <div className="flex flex-col gap-2 pt-2 border-t border-gray-100 dark:border-gray-800">
                    <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex justify-between">
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
                    <p className="text-xs text-gray-500 dark:text-gray-400">Delay before accepting a newly detected token to prevent mid-air bugs.</p>
                  </div>
                  
                  <div className="flex flex-col gap-2 pt-2 border-t border-gray-100 dark:border-gray-800">
                    <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex justify-between">
                      <span>NFC Timeout</span>
                      <span className="text-brand-green">{nfcTimeout.toFixed(1)}s</span>
                    </label>
                    <input 
                      type="range" 
                      min="1.0" max="60.0" step="1.0" 
                      value={nfcTimeout}
                      onChange={(e) => setNfcTimeout(parseFloat(e.target.value))}
                      className="w-full accent-brand-green"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400">Time before a registered NFC tag is reset.</p>
                  </div>
                  
                  {simulationMode && aiEnabled && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 dark:bg-amber-950 dark:border-amber-900 dark:text-amber-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
                      <span className="font-bold">⚠</span>
                      <p>AI Advisory active. The AI will advise you via the UI, but you must physically place its black token.</p>
                    </div>
                  )}
                  {simulationMode && !aiEnabled && (
                    <div className="bg-blue-50 border border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-900 dark:text-blue-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
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
