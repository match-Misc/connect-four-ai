import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { cn } from '../utils';

interface DrawProps {
  /** Mounts the overlay and starts the animation. */
  active: boolean;
  /** How long the overlay stays up, in ms. Includes the fade-out. */
  duration?: number;
  /** Fired once the overlay has faded out. */
  onDone?: () => void;
}

/** Same warm palette as the loss screen — a draw is a friendly ending too. */
const TOKEN_COLORS = ['#B1CA21', '#FFD93D', '#4ECDC4', '#FF9F1C', '#FFFFFF'];

/** Low on purpose: the exhibition display shares its CPU with the vision thread. */
const TOKEN_COUNT = 20;

/** How long the overlay spends fading out at the end. */
const FADE_MS = 600;

const rand = (min: number, max: number) => min + Math.random() * (max - min);

/**
 * The end screen for a full board with no four in a row.
 *
 * Like the loss screen this is CSS-only rather than canvas-driven: the scene is
 * a handful of transforms, so the compositor carries it and the main thread
 * stays free for the vision loop. Neither side lost here, so both tokens are on
 * screen and neither is bigger than the other.
 */
export function Draw({ active, duration = 7000, onDone }: DrawProps) {
  const [leaving, setLeaving] = useState(false);
  // Held in a ref so a changing callback identity doesn't restart the timers.
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  // Keyed on `active` so every showing re-rolls: two draws in a row should not
  // put the tokens in the same places.
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
      {/* Dims the board so the two tokens carry the moment. The game is over,
          and the overlay passes clicks through, so Reset stays reachable. */}
      <div className="absolute inset-0 bg-gray-950/55 animate-overlay-scrim" />

      {/* Tokens drift upwards: rising reads as hopeful. */}
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
        <TokenHighFive />

        {/* Both players are addressed together — nobody is being told off. */}
        <div className="animate-overlay-text relative max-w-3xl rounded-[2rem] border-4 border-brand-green bg-white dark:bg-gray-900 px-8 py-6 lg:px-14 lg:py-10 shadow-2xl text-center">
          <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-7 h-7 rotate-45 border-l-4 border-t-4 border-brand-green bg-white dark:bg-gray-900" />
          <p className="font-black tracking-tight text-gray-800 dark:text-gray-100 text-3xl sm:text-4xl lg:text-5xl">
            Unentschieden!
          </p>
          <p className="mt-3 lg:mt-5 font-bold text-brand-green text-2xl sm:text-3xl lg:text-4xl">
            Ihr wart beide gleich gut!
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * The two playing pieces bumping each other, sized by their container. Their
 * colours are the board's own token colours, so a child can tell at a glance
 * that this is "you and the robot", not two new characters.
 */
function TokenHighFive() {
  return (
    <svg
      viewBox="0 0 240 160"
      aria-hidden="true"
      className="w-72 h-48 sm:w-96 sm:h-64 lg:w-[30rem] lg:h-80 drop-shadow-[0_10px_30px_rgba(0,0,0,0.45)]"
    >
      {/* Human token. Pivots around its own centre, so the tilt in the bump
          keyframes reads as a lean rather than a swing. */}
      <g
        className="animate-draw-bump-left"
        style={{ transformBox: 'view-box', transformOrigin: '72px 80px' }}
      >
        <circle cx="72" cy="80" r="44" className="fill-brand-green" />
        <circle
          cx="72"
          cy="80"
          r="38"
          fill="none"
          strokeWidth="5"
          stroke="rgba(255,255,255,0.35)"
        />
        <g className="animate-eye-blink" style={{ transformBox: 'fill-box', transformOrigin: 'center' }}>
          <circle cx="58" cy="70" r="6" className="fill-slate-900" />
          <circle cx="86" cy="70" r="6" className="fill-slate-900" />
        </g>
        <path
          d="M56 94 Q72 108 88 94"
          strokeWidth="6"
          strokeLinecap="round"
          fill="none"
          className="stroke-slate-900"
        />
      </g>

      {/* Robot token. */}
      <g
        className="animate-draw-bump-right"
        style={{ transformBox: 'view-box', transformOrigin: '168px 80px' }}
      >
        <circle cx="168" cy="80" r="44" className="fill-gray-900 dark:fill-black" />
        <circle
          cx="168"
          cy="80"
          r="38"
          fill="none"
          strokeWidth="5"
          stroke="rgba(255,255,255,0.35)"
        />
        <g className="animate-eye-blink" style={{ transformBox: 'fill-box', transformOrigin: 'center' }}>
          <circle cx="154" cy="70" r="6" className="fill-[#4ECDC4]" />
          <circle cx="182" cy="70" r="6" className="fill-[#4ECDC4]" />
        </g>
        <path
          d="M152 94 Q168 108 184 94"
          strokeWidth="6"
          strokeLinecap="round"
          fill="none"
          className="stroke-[#4ECDC4]"
        />
      </g>

      {/* The clink, on the beat the two tokens meet. */}
      <g
        className="animate-draw-spark"
        style={{ transformBox: 'fill-box', transformOrigin: 'center' }}
      >
        <path
          d="M120 54 L126 74 L146 80 L126 86 L120 106 L114 86 L94 80 L114 74 Z"
          className="fill-[#FFD93D]"
        />
      </g>
    </svg>
  );
}
