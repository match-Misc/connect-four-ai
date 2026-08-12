import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { cn } from '../utils';

interface RobotWinsProps {
  /** Mounts the overlay and starts the animation. */
  active: boolean;
  /** How long the overlay stays up, in ms. Includes the fade-out. */
  duration?: number;
  /** Fired once the overlay has faded out. */
  onDone?: () => void;
}

/** Warm tones only — a loss screen for kids should read as "try again", not as an alarm. */
const TOKEN_COLORS = ['#B1CA21', '#FFD93D', '#4ECDC4', '#FF9F1C', '#FFFFFF'];

/** Low on purpose: the exhibition display shares its CPU with the vision thread. */
const TOKEN_COUNT = 22;

/** How long the overlay spends fading out at the end. */
const FADE_MS = 600;

const rand = (min: number, max: number) => min + Math.random() * (max - min);

/**
 * The consolation screen for a robot win.
 *
 * Everything here is CSS-driven rather than canvas-driven (unlike the
 * fireworks): the whole scene is a handful of transforms, so the compositor
 * carries it and the main thread stays free for the vision loop.
 */
export function RobotWins({ active, duration = 7000, onDone }: RobotWinsProps) {
  const [leaving, setLeaving] = useState(false);
  // Held in a ref so a changing callback identity doesn't restart the timers.
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  // Keyed on `active` so every showing re-rolls: two losses in a row should
  // not put the tokens in the same places.
  const tokens = useMemo(
    () =>
      Array.from({ length: TOKEN_COUNT }, (_, i) => ({
        left: rand(2, 96),
        size: rand(14, 34),
        // Negative delays start the loop mid-flight, so the screen is already
        // populated on the first frame instead of filling from the bottom.
        delay: -rand(0, 7),
        duration: rand(6, 10),
        drift: rand(-50, 50),
        color: TOKEN_COLORS[i % TOKEN_COLORS.length],
      })),
    // `active` is the re-roll trigger, not an input to the layout.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [active],
  );

  useEffect(() => {
    if (!active) {
      setLeaving(false);
      return;
    }
    const fade = setTimeout(() => setLeaving(true), Math.max(0, duration - FADE_MS));
    const done = setTimeout(() => onDoneRef.current?.(), duration);
    return () => {
      clearTimeout(fade);
      clearTimeout(done);
    };
  }, [active, duration]);

  if (!active) return null;

  return (
    <div
      className={cn(
        'fixed inset-0 z-[100] flex flex-col items-center justify-center gap-8 px-6 pointer-events-none',
        leaving && 'animate-overlay-out',
      )}
    >
      {/* Dims the board so the robot carries the moment. The game is over, and
          the overlay passes clicks through, so the header controls stay
          reachable. */}
      <div className="absolute inset-0 bg-gray-950/55 animate-overlay-scrim" />

      {/* Tokens drift upwards: rising reads as hopeful, falling reads as defeat. */}
      <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
        {tokens.map((token, i) => (
          <span
            key={i}
            className="absolute -bottom-16 rounded-full animate-token-float"
            style={{
              left: `${token.left}%`,
              width: token.size,
              height: token.size,
              backgroundColor: token.color,
              boxShadow: 'inset 0 0 0 3px rgba(255,255,255,0.35)',
              animationDuration: `${token.duration}s`,
              animationDelay: `${token.delay}s`,
              '--drift': `${token.drift}px`,
            } as CSSProperties}
          />
        ))}
      </div>

      <div className="relative flex flex-col items-center gap-8 animate-overlay-in">
        <div className="animate-robot-bob">
          <RobotMascot />
        </div>

        {/* Speech bubble — the robot is talking to the child, not gloating at them. */}
        <div className="animate-overlay-text relative max-w-3xl rounded-[2rem] border-4 border-brand-green bg-white dark:bg-gray-900 px-8 py-6 lg:px-14 lg:py-10 shadow-2xl text-center">
          <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-7 h-7 rotate-45 border-l-4 border-t-4 border-brand-green bg-white dark:bg-gray-900" />
          <p className="font-black tracking-tight text-gray-800 dark:text-gray-100 text-3xl sm:text-4xl lg:text-5xl">
            Du hast leider verloren.
          </p>
          <p className="mt-3 lg:mt-5 font-bold text-brand-green text-2xl sm:text-3xl lg:text-4xl">
            Versuche es in der nächsten Runde nochmal.
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * A cartoon robot, sized by its container. Kept deliberately round and
 * wide-eyed — the same machine that just won the game has to look like a
 * playmate here.
 */
function RobotMascot() {
  return (
    <svg
      viewBox="0 0 200 200"
      aria-hidden="true"
      className="w-48 h-48 sm:w-60 sm:h-60 lg:w-72 lg:h-72 drop-shadow-[0_10px_30px_rgba(0,0,0,0.45)]"
    >
      {/* Antenna */}
      <path
        d="M100 44 V30"
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
        className="stroke-slate-700 dark:stroke-slate-200"
      />
      <circle cx="100" cy="22" r="8" className="fill-[#FFD93D] animate-robot-glow" />

      {/* Ears */}
      <rect x="45" y="62" width="11" height="26" rx="5.5" className="fill-slate-700 dark:fill-slate-200" />
      <rect x="144" y="62" width="11" height="26" rx="5.5" className="fill-slate-700 dark:fill-slate-200" />

      {/* Head and face screen */}
      <rect x="56" y="42" width="88" height="66" rx="22" className="fill-slate-700 dark:fill-slate-200" />
      <rect x="68" y="54" width="64" height="42" rx="14" className="fill-slate-900 dark:fill-slate-600" />

      {/* Eyes. Squashed vertically for the blink, so the origin has to sit on
          the group's own box rather than the viewBox. */}
      <g
        className="animate-eye-blink"
        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
      >
        <circle cx="88" cy="70" r="7" className="fill-[#4ECDC4]" />
        <circle cx="112" cy="70" r="7" className="fill-[#4ECDC4]" />
      </g>
      <path
        d="M87 82 Q100 93 113 82"
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
        className="stroke-[#4ECDC4]"
      />

      {/* Neck and body */}
      <rect x="92" y="104" width="16" height="12" className="fill-slate-700 dark:fill-slate-200" />
      <rect x="56" y="114" width="88" height="62" rx="20" className="fill-slate-700 dark:fill-slate-200" />
      <circle cx="100" cy="142" r="10" className="fill-brand-green animate-robot-glow" />

      {/* Resting arm */}
      <path
        d="M58 134 L38 152"
        strokeWidth="11"
        strokeLinecap="round"
        fill="none"
        className="stroke-slate-700 dark:stroke-slate-200"
      />
      <circle cx="34" cy="156" r="9" className="fill-slate-700 dark:fill-slate-200" />

      {/* Waving arm. Pivots on the shoulder, so the origin is given in viewBox
          units instead of the group's own box. */}
      <g
        className="animate-robot-wave"
        style={{ transformBox: 'view-box', transformOrigin: '142px 130px' }}
      >
        <path
          d="M142 130 L166 106"
          strokeWidth="11"
          strokeLinecap="round"
          fill="none"
          className="stroke-slate-700 dark:stroke-slate-200"
        />
        <circle cx="170" cy="102" r="9" className="fill-slate-700 dark:fill-slate-200" />
      </g>

      {/* Feet */}
      <rect x="68" y="172" width="26" height="13" rx="6.5" className="fill-slate-700 dark:fill-slate-200" />
      <rect x="106" y="172" width="26" height="13" rx="6.5" className="fill-slate-700 dark:fill-slate-200" />
    </svg>
  );
}
