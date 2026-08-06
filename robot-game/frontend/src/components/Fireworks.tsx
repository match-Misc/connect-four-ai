import { useEffect, useRef } from 'react';

interface FireworksProps {
  /** Mounts the canvas and starts the show. */
  active: boolean;
  /** How long to keep launching new shells, in ms. */
  duration?: number;
  /** Fired once the last spark has faded. */
  onDone?: () => void;
}

const COLORS = ['#B1CA21', '#FFD93D', '#FF9F1C', '#FF5E5B', '#4ECDC4', '#FFFFFF'];

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** 1 at birth, 0 when dead. */
  life: number;
  decay: number;
  color: string;
}

interface Shell {
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
}

const rand = (min: number, max: number) => min + Math.random() * (max - min);
const pick = <T,>(items: T[]): T => items[Math.floor(Math.random() * items.length)];

/**
 * Runs the show on a canvas until every spark has faded, then calls `onDone`.
 * Returns a stop handle.
 *
 * Sparks are drawn as plain squares and the frame count is kept low on
 * purpose — the exhibition display shares its CPU with the vision thread, and
 * per-particle arcs and gradients cost more than they add visually.
 */
function runFireworks(
  canvas: HTMLCanvasElement,
  { duration = 9000, onDone }: { duration?: number; onDone?: () => void } = {},
): () => void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return () => {};

  let width = 0;
  let height = 0;
  // Gravity scales with the viewport so the show looks the same on a laptop
  // and on the exhibition display.
  let gravity = 0;

  const resize = () => {
    width = window.innerWidth;
    height = window.innerHeight;
    // Deliberately rendered at 1x: the sparks are soft blobs, so the extra
    // pixels of a retina buffer only cost frame time.
    canvas.width = width;
    canvas.height = height;
    gravity = height * 1.05;
  };
  resize();
  window.addEventListener('resize', resize);

  const shells: Shell[] = [];
  const particles: Particle[] = [];

  const launch = () => {
    const rise = rand(0.48, 0.72) * height;
    shells.push({
      // Kept away from the edges so the wider bursts stay on screen.
      x: rand(0.2, 0.8) * width,
      y: height + 10,
      vx: rand(-0.05, 0.05) * width,
      vy: -Math.sqrt(2 * gravity * rise),
      color: pick(COLORS),
    });
  };

  const explode = (shell: Shell) => {
    // Count rises with the radius so the wider bursts don't read as sparse.
    const count = Math.floor(rand(80, 120));
    const power = rand(0.5, 1.00) * height;
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      // sqrt spreads the sparks evenly across the disc instead of clumping
      // them at the rim.
      const speed = power * Math.sqrt(Math.random()) * rand(0.75, 1.15);
      particles.push({
        x: shell.x,
        y: shell.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        // Slightly slower burn, so the sparks reach the wider radius before
        // they fade out.
        decay: rand(0.34, 0.9),
        color: shell.color,
      });
    }
  };

  const start = performance.now();
  let last = start;
  let nextLaunch = start;
  let frame = 0;
  let stopped = false;

  const cleanup = () => {
    stopped = true;
    cancelAnimationFrame(frame);
    window.removeEventListener('resize', resize);
  };

  const tick = (now: number) => {
    if (stopped) return;
    frame = requestAnimationFrame(tick);
    // Clamp so a dropped frame doesn't teleport everything.
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    const elapsed = now - start;
    const launching = elapsed < duration;

    if (launching && now >= nextLaunch) {
      launch();
      if (Math.random() < 0.3) launch();
      nextLaunch = now + rand(420, 800);
    }

    // Fade the previous frame rather than clearing it: this leaves the sparks'
    // trails while keeping the UI underneath untouched.
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'lighter';

    for (let i = shells.length - 1; i >= 0; i--) {
      const shell = shells[i];
      shell.vy += gravity * dt;
      shell.x += shell.vx * dt;
      shell.y += shell.vy * dt;
      // Apex reached — burst.
      if (shell.vy >= 0) {
        explode(shell);
        shells.splice(i, 1);
        continue;
      }
      ctx.fillStyle = shell.color;
      ctx.fillRect(shell.x - 2, shell.y - 2, 4, 4);
    }

    const drag = Math.pow(0.955, dt * 60);
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.vx *= drag;
      p.vy = p.vy * drag + gravity * 0.34 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= p.decay * dt;
      if (p.life <= 0 || p.y > height + 40) {
        particles.splice(i, 1);
        continue;
      }
      const size = 4.5 + p.life * 3.5;
      ctx.globalAlpha = p.life;
      ctx.fillStyle = p.color;
      ctx.fillRect(p.x - size / 2, p.y - size / 2, size, size);
    }

    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';

    if (!launching && !shells.length && !particles.length) {
      cleanup();
      ctx.clearRect(0, 0, width, height);
      onDone?.();
    }
  };

  frame = requestAnimationFrame(tick);
  return cleanup;
}

export function Fireworks({ active, duration = 9000, onDone }: FireworksProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Held in a ref so a changing callback identity doesn't restart the show.
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    if (!active || !canvasRef.current) return;
    return runFireworks(canvasRef.current, {
      duration,
      onDone: () => onDoneRef.current?.(),
    });
  }, [active, duration]);

  if (!active) return null;

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="fixed inset-0 w-screen h-screen z-[100] pointer-events-none"
    />
  );
}
