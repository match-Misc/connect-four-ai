import { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { cn } from './utils';
import { ConnectFourGrid } from './components/ConnectFourGrid';
import { Draw } from './components/Draw';
import { Fireworks } from './components/Fireworks';
import { RobotWins } from './components/RobotWins';
import { ThemeToggle } from './components/ThemeToggle';
import { Activity, Bot, User, Settings2, Bug, Gamepad2, Sparkles, Nfc, Handshake } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000/api`;

/** Shared empty value, so "no invalid stones" is the same array every poll. */
const NO_STONES: number[][] = [];

/** Readable names for the backend's robot_state values. */
const ROBOT_STATE_LABELS: Record<string, string> = {
  idle: 'Bereit',
  analyzing: 'Analysiert',
  thinking: 'Denkt nach',
  moving: 'Bewegt sich',
  waiting_for_drop: 'Wartet auf Einwurf',
};

/** The backend's difficulty values with the labels shown on the buttons. Only
    the labels are German — the values travel to the backend as they are. */
const DIFFICULTIES: { value: string; label: string }[] = [
  { value: 'easy', label: 'Leicht' },
  { value: 'medium', label: 'Mittel' },
  { value: 'hard', label: 'Schwer' },
  { value: 'impossible', label: 'Unmöglich' },
];

/** Deep equality for the small number grids the backend sends. */
function sameGrid(a: number[][], b: number[][]) {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let r = 0; r < a.length; r++) {
    if (a[r].length !== b[r].length) return false;
    for (let c = 0; c < a[r].length; c++) {
      if (a[r][c] !== b[r][c]) return false;
    }
  }
  return true;
}

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
  // Robot handshake signals, straight from the pendant: whether a grab code is
  // outstanding, whether the robot said it holds a token, and how many times it
  // has dialled in this session.
  const [grabRequested, setGrabRequested] = useState<boolean>(false);
  const [stoneHeld, setStoneHeld] = useState<boolean>(false);
  const [robotConnects, setRobotConnects] = useState<number>(0);
  const [celebrating, setCelebrating] = useState<boolean>(false);
  const [consoling, setConsoling] = useState<boolean>(false);
  const [tied, setTied] = useState<boolean>(false);

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
  // Clicking a difficulty sets it locally and POSTs it, but the board-state poll
  // runs every 150ms: a response that left the backend before the POST landed
  // still carries the old difficulty, and it used to overwrite the click — which
  // then re-fired the config effect and posted the old value back, losing the
  // change for good. So a difficulty we just chose is held until the backend
  // echoes it. The robot's toggle button still comes through: the hold only
  // covers our own in-flight change, and it expires regardless, so a POST that
  // never landed cannot freeze the UI on a value the backend does not have.
  const pendingDifficulty = useRef<{ value: string; until: number } | null>(null);

  const fetchBoardState = async () => {
    try {
      const res = await fetch(`${API_BASE}/board-state`);
      const data = await res.json();
      // Reusing the previous array when nothing moved keeps React from
      // re-rendering the grid on every poll. That matters most during the
      // end-of-game animations: the board is frozen then, so this leaves the
      // main thread to the fireworks instead of reconciling 42 unchanged cells
      // several times a second.
      if (data.board) setBoard(prev => (sameGrid(prev, data.board) ? prev : data.board));
      setTurn(data.turn);
      setRobotState(data.robot_state);
      setSimulationMode(data.simulation_mode);
      setGameOver(data.game_over || false);
      // `??`, not `||`: a draw is winner 0, which `||` would flatten into the
      // same null the backend sends while a game is still running.
      setWinner(data.winner ?? null);
      setErrorMsg(data.error_msg || null);
      const stones = data.invalid_stones || NO_STONES;
      setInvalidStones(prev => (sameGrid(prev, stones) ? prev : stones));
      if (data.debounce_time !== undefined) {
        setDebounceTime(data.debounce_time);
      }
      if (data.ai_enabled !== undefined) {
        setAiEnabled(data.ai_enabled);
      }
      if (data.difficulty !== undefined) {
        const pending = pendingDifficulty.current;
        if (pending && data.difficulty !== pending.value && Date.now() < pending.until) {
          // Stale response from before our POST landed; keep showing the click.
        } else {
          pendingDifficulty.current = null;
          setDifficulty(data.difficulty);
        }
      }
      if (data.tcp_connected !== undefined) {
        setTcpConnected(data.tcp_connected);
      }
      if (data.nfc_connected !== undefined) {
        setNfcConnected(data.nfc_connected);
      }
      if (data.grab_requested !== undefined) {
        setGrabRequested(data.grab_requested);
      }
      if (data.stone_held !== undefined) {
        setStoneHeld(data.stone_held);
      }
      if (data.robot_connects !== undefined) {
        setRobotConnects(data.robot_connects);
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
    const interval = setInterval(fetchBoardState, celebrating || consoling || tied ? 600 : 150);
    return () => clearInterval(interval);
  }, [celebrating, consoling, tied]);

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
    } else if (winner === 0) {
      gameOverAnimated.current = true;
      setTied(true);
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
      // The difficulty in that body never reached the backend, so stop holding
      // the polled value back instead of waiting out the expiry.
      pendingDifficulty.current = null;
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
      <Draw active={tied} onDone={() => setTied(false)} />
      {celebrating && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center pointer-events-none">
          <h2 className="animate-win-text text-center font-black uppercase tracking-tight text-brand-green text-5xl sm:text-7xl lg:text-8xl drop-shadow-[0_4px_20px_rgba(0,0,0,0.6)]">
            Du hast gewonnen!
          </h2>
        </div>
      )}
      <div className="w-full px-4 xl:px-8 flex flex-col gap-4 h-full">

        {/* Header */}
        {/* Stacked rather than side-by-side: the difficulty buttons and the turn
            indicators are the two things a visitor has to hit or read from a
            step away, so each gets the full width of the card to grow into. */}
        <header className="relative flex flex-col items-center bg-white dark:bg-gray-900 p-4 lg:p-5 portrait:p-4 pr-14 lg:pr-16 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 gap-4 portrait:gap-3 shrink-0">
          <ThemeToggle className="absolute top-3 right-3" />
          <div className="flex items-center gap-5">
            <img src="/match_NUR_Logo_10pt.svg" alt="match-Logo" className="h-20 lg:h-28" onError={(e) => (e.currentTarget.style.display = 'none')} />
            <div className="flex flex-col">
              <h1 className="text-4xl lg:text-6xl font-bold text-gray-800 dark:text-gray-100 tracking-tight whitespace-nowrap">Vier Gewinnt KI</h1>
              <div className="flex gap-5 mt-2 items-center">
                <Link to="/game" className={cn("text-base lg:text-lg font-semibold flex items-center gap-1.5 transition-colors", !showDebug ? "text-brand-green" : "text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300")}>
                  <Gamepad2 size={20} /> Spielen
                </Link>
                <Link to={showDebug ? "/game" : "/debugging"} title={showDebug ? "Debug-Ansicht schließen" : "Debug-Ansicht öffnen"} className={cn("text-base lg:text-lg font-semibold flex items-center gap-1.5 transition-colors", showDebug ? "text-purple-600 dark:text-purple-400" : "text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300")}>
                  <Bug size={20} /> Debug
                </Link>
                <div className="h-4 w-px bg-gray-300 dark:bg-gray-700 mx-1" />
                <div className="flex items-center gap-1.5" title={tcpConnected ? "Roboter verbunden" : "Roboter getrennt"}>
                  <RobotArmIcon size={16} className="text-gray-500 dark:text-gray-400" />
                  <span 
                    className={cn(
                      "w-2 h-2 rounded-full",
                      tcpConnected ? "bg-green-500" : "bg-red-500"
                    )}
                  />
                </div>
                <div className="flex items-center gap-1.5 ml-2" title={nfcConnected ? "NFC verbunden" : "NFC getrennt"}>
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

          {/* Difficulty Selector */}
          <div className="flex flex-wrap justify-center gap-1 bg-gray-100 dark:bg-gray-800 p-2 rounded-2xl">
            {DIFFICULTIES.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => {
                  pendingDifficulty.current = { value, until: Date.now() + 2000 };
                  setDifficulty(value);
                }}
                className={cn(
                  "px-6 py-3.5 lg:px-10 lg:py-5 rounded-xl text-xl lg:text-3xl font-black transition-all",
                  difficulty === value ? "bg-white dark:bg-gray-700 shadow-md text-brand-green scale-105" : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                )}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex items-center w-full mt-2 lg:mt-4">
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
                {nfcOverwritten ? "Überschrieben" : nfcData ? "NFC registriert" : "NFC-Tag"}
              </div>
            </div>

            {/* Center aligned items. Whose turn it is has to be readable from a
                step back, so these are the largest elements on the page. The
                border width is shared by both states — only its colour changes,
                so the inactive field does not jump when it lights up. */}
            <div className="flex items-center gap-6 lg:gap-12">
              <div className={cn(
                "flex items-center gap-5 px-10 py-6 lg:px-14 lg:py-8 rounded-3xl font-black text-4xl lg:text-6xl uppercase tracking-wide whitespace-nowrap transition-all duration-300 border-4 lg:border-[6px]",
                turn === 'human'
                  ? "bg-brand-green/20 text-green-900 dark:text-brand-green border-brand-green scale-105 shadow-lg"
                  : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600 border-transparent opacity-60"
              )}>
                <User className="w-12 h-12 lg:w-16 lg:h-16" /> Mensch
              </div>
              <div className={cn(
                "flex items-center gap-5 px-10 py-6 lg:px-14 lg:py-8 rounded-3xl font-black text-4xl lg:text-6xl uppercase tracking-wide whitespace-nowrap transition-all duration-300 border-4 lg:border-[6px]",
                turn === 'robot'
                  ? "bg-gray-800 text-white dark:bg-gray-100 dark:text-gray-900 border-gray-800 dark:border-gray-100 scale-105 shadow-lg"
                  : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600 border-transparent opacity-60"
              )}>
                <Bot className="w-12 h-12 lg:w-16 lg:h-16" /> Roboter
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
              <h2 className="text-2xl lg:text-3xl font-black text-gray-800 dark:text-gray-100 uppercase tracking-widest mb-1 lg:mb-2">Spiel beendet!</h2>
              <div className="text-lg lg:text-xl font-bold">
                {winner === 1 ? (
                  <span className="text-brand-green flex items-center gap-2">🎉 Mensch gewinnt! 🎉</span>
                ) : winner === 2 ? (
                  <span className="text-red-500 flex items-center gap-2">🤖 Roboter gewinnt! 🤖</span>
                ) : (
                  <span className="text-gray-500 flex items-center gap-2">🤝 Unentschieden! 🤝</span>
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
                  <Activity size={20} className="text-purple-500" /> Kamerabild
                </h2>
                <div className="text-sm text-gray-600 dark:text-gray-400 flex gap-4">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-white dark:border-gray-600 inline-block bg-gray-200 dark:bg-gray-700"></span> Leer</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-green-500 inline-block bg-transparent"></span> Spieler 1 erkannt</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-red-500 inline-block bg-transparent"></span> Spieler 2 erkannt</span>
                </div>
                <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 aspect-[4/3] flex items-center justify-center relative w-full">
                  <img 
                    src={`${API_BASE.replace('/api', '')}/api/video-feed`} 
                    alt="RealSense-Stream"
                    className="absolute top-0 left-0 w-full h-full object-cover"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none';
                      e.currentTarget.parentElement!.innerHTML = '<span class="text-xs text-gray-500">Kamerabild offline</span>';
                    }}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-6">
                <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <Activity size={20} className="text-blue-500" /> Roboter-Status
                  </h2>

                  <div className="flex flex-col gap-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">Zustand</span>
                      <span className={cn(
                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                        robotState === 'idle' ? "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300" :
                        robotState === 'analyzing' ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 animate-pulse" :
                        robotState === 'thinking' ? "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300 animate-pulse" :
                        robotState === 'waiting_for_drop' ? "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 animate-pulse" :
                        "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300 animate-bounce"
                      )}>
                        {ROBOT_STATE_LABELS[robotState] ?? robotState}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">Verbindung</span>
                      <span className={cn(
                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                        simulationMode ? "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300" :
                        tcpConnected ? "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300" :
                        "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300"
                      )}>
                        {simulationMode ? "Simuliert" : tcpConnected ? "Verbunden" : "Getrennt"}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">Greifer</span>
                      <span className={cn(
                        "px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider",
                        simulationMode ? "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300" :
                        stoneHeld ? "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300" :
                        grabRequested ? "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 animate-pulse" :
                        "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300"
                      )}>
                        {simulationMode ? "—" : stoneHeld ? "Stein gegriffen" : grabRequested ? "Greifbefehl gesendet…" : "Leer"}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">Zielspalte</span>
                      <span className="text-sm font-bold text-gray-800 dark:text-gray-100">
                        {robotTargetCol !== null ? robotTargetCol + 1 : '—'}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500 dark:text-gray-400 font-bold">Verbindungsaufbauten</span>
                      <span className="text-sm font-bold text-gray-800 dark:text-gray-100">{robotConnects}</span>
                    </div>
                  </div>

                  {/* The move is parked in run_robot_move() until the pendant acks the
                      grab, so this pairing is the one stall the panel can name outright. */}
                  {!simulationMode && grabRequested && !stoneHeld && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 dark:bg-amber-950 dark:border-amber-900 dark:text-amber-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2">
                      <span className="font-bold">⚠</span>
                      <p>Greifbefehl gesendet – es wird darauf gewartet, dass der Roboter den Stein bestätigt. Bis dahin wird die berechnete Spalte zurückgehalten.</p>
                    </div>
                  )}
                  {!simulationMode && !tcpConnected && (
                    <div className="bg-red-50 border border-red-200 text-red-800 dark:bg-red-950 dark:border-red-900 dark:text-red-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2">
                      <span className="font-bold">⚠</span>
                      <p>Kein Bedienpanel verbunden. Der Roboter verbindet sich selbst zum Backend – starte sein Programm, um die Verbindung wiederherzustellen.</p>
                    </div>
                  )}
                </div>

                <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <Settings2 size={20} className="text-gray-500" /> Einstellungen
                  </h2>

                  <label className="flex items-center justify-between cursor-pointer group">
                    <span className="text-sm font-bold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">KI aktiviert</span>
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
                    <span className="text-sm font-bold text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white transition-colors">Simulationsmodus</span>
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
                      <span>CV-Entprellzeit</span>
                      <span className="text-brand-green">{debounceTime.toFixed(1)}s</span>
                    </label>
                    <input 
                      type="range" 
                      min="0.1" max="3.0" step="0.1" 
                      value={debounceTime}
                      onChange={(e) => setDebounceTime(parseFloat(e.target.value))}
                      className="w-full accent-brand-green"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400">Wartezeit, bevor ein neu erkannter Stein übernommen wird – verhindert Fehler bei fallenden Steinen.</p>
                  </div>
                  
                  <div className="flex flex-col gap-2 pt-2 border-t border-gray-100 dark:border-gray-800">
                    <label className="text-sm font-bold text-gray-700 dark:text-gray-300 flex justify-between">
                      <span>NFC-Timeout</span>
                      <span className="text-brand-green">{nfcTimeout.toFixed(1)}s</span>
                    </label>
                    <input 
                      type="range" 
                      min="1.0" max="60.0" step="1.0" 
                      value={nfcTimeout}
                      onChange={(e) => setNfcTimeout(parseFloat(e.target.value))}
                      className="w-full accent-brand-green"
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400">Zeit, bis ein registrierter NFC-Tag zurückgesetzt wird.</p>
                  </div>
                  
                  {simulationMode && aiEnabled && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 dark:bg-amber-950 dark:border-amber-900 dark:text-amber-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
                      <span className="font-bold">⚠</span>
                      <p>KI-Beratung aktiv. Die KI zeigt ihren Zug in der Oberfläche an, den schwarzen Stein musst du aber selbst einwerfen.</p>
                    </div>
                  )}
                  {simulationMode && !aiEnabled && (
                    <div className="bg-blue-50 border border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-900 dark:text-blue-300 text-xs px-3 py-2 rounded-lg flex items-start gap-2 mt-2">
                      <span className="font-bold">ℹ</span>
                      <p>Manueller 2-Spieler-Modus. Ihr spielt komplett von Hand, ohne Eingriff der KI.</p>
                    </div>
                  )}
                </div>

                <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 flex flex-col gap-4">
                  <h2 className="text-lg font-bold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                    <Bug size={20} className="text-purple-500" /> Test
                  </h2>

                  <p className="text-xs text-gray-500 dark:text-gray-400">Die Schluss-Animationen ansehen, ohne ein Spiel zu spielen.</p>

                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        setCelebrating(false);
                        // Remount on the next frame so a second click restarts the show.
                        requestAnimationFrame(() => setCelebrating(true));
                      }}
                      title="Sieg-Animation ansehen"
                      className="flex-1 bg-amber-100 hover:bg-amber-200 text-amber-700 dark:bg-amber-950 dark:hover:bg-amber-900 dark:text-amber-300 px-4 py-2.5 rounded-xl text-sm font-bold transition-colors shadow-sm flex items-center justify-center gap-2"
                    >
                      <Sparkles className="w-4 h-4" /> Sieg
                    </button>
                    <button
                      onClick={() => {
                        setConsoling(false);
                        requestAnimationFrame(() => setConsoling(true));
                      }}
                      title="Niederlage-Animation ansehen"
                      className="flex-1 bg-sky-100 hover:bg-sky-200 text-sky-700 dark:bg-sky-950 dark:hover:bg-sky-900 dark:text-sky-300 px-4 py-2.5 rounded-xl text-sm font-bold transition-colors shadow-sm flex items-center justify-center gap-2"
                    >
                      <Bot className="w-4 h-4" /> Niederlage
                    </button>
                    <button
                      onClick={() => {
                        setTied(false);
                        requestAnimationFrame(() => setTied(true));
                      }}
                      title="Unentschieden-Animation ansehen"
                      className="flex-1 bg-emerald-100 hover:bg-emerald-200 text-emerald-700 dark:bg-emerald-950 dark:hover:bg-emerald-900 dark:text-emerald-300 px-4 py-2.5 rounded-xl text-sm font-bold transition-colors shadow-sm flex items-center justify-center gap-2"
                    >
                      <Handshake className="w-4 h-4" /> Unentschieden
                    </button>
                  </div>
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
