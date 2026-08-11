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

/**
 * Sparks are drawn in batches of one colour at one opacity, so opacity is
 * quantised into this many steps. Twelve is past the point where the banding
 * is visible on a fading spark.
 */
const ALPHA_STEPS = 12;
const BUCKETS = COLORS.length * ALPHA_STEPS;

/** One `rgba()` string per (colour, alpha step). Built once, never at frame time. */
const FILL_STYLES = COLORS.flatMap((hex) => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return Array.from(
    { length: ALPHA_STEPS },
    (_, step) => `rgba(${r},${g},${b},${((step + 0.5) / ALPHA_STEPS).toFixed(3)})`,
  );
});

/** Hard ceilings, so the pools can be allocated up front and never grow. */
const MAX_PARTICLES = 1600;
const MAX_SHELLS = 16;

/**
 * The canvas never gets more pixels than this. On the exhibition display a 1:1
 * buffer means clearing several million pixels every frame for sparks that are
 * soft blobs anyway — the browser scales the smaller buffer back up for free.
 */
const MAX_CANVAS_PIXELS = 2_100_000;

/** Fraction of the trail left standing after one second, at the reference 60fps. */
const TRAIL_KEEP_PER_FRAME = 0.7;

const rand = (min: number, max: number) => min + Math.random() * (max - min);

/**
 * Runs the show on a canvas until every spark has faded, then calls `onDone`.
 * Returns a stop handle.
 *
 * Written to keep every frame the same length rather than to be clever: the
 * exhibition display shares its CPU with the vision thread, and an animation
 * that hitches reads worse than one with fewer sparks. Hence the flat typed
 * arrays (no per-spark objects for the GC to collect mid-show), the batched
 * fills (one canvas state change per colour instead of one per spark), and the
 * spark budget that gives way when frames start running long.
 */
function runFireworks(
  canvas: HTMLCanvasElement,
  { duration = 9000, onDone }: { duration?: number; onDone?: () => void } = {},
): () => void {
  const ctx = canvas.getContext('2d', { alpha: true });
  if (!ctx) return () => {};

  let width = 0;
  let height = 0;
  // Gravity scales with the viewport so the show looks the same on a laptop
  // and on the exhibition display.
  let gravity = 0;
  // Buffer pixels per CSS pixel. Only ever below 1, on very large displays.
  let renderScale = 1;

  const resize = () => {
    const cssWidth = window.innerWidth;
    const cssHeight = window.innerHeight;
    renderScale = Math.min(1, Math.sqrt(MAX_CANVAS_PIXELS / (cssWidth * cssHeight)));
    const next = {
      w: Math.max(1, Math.round(cssWidth * renderScale)),
      h: Math.max(1, Math.round(cssHeight * renderScale)),
    };
    if (next.w === width && next.h === height) return;

    // Carry the sparks over to the new geometry, so a resize mid-show doesn't
    // strand them off screen.
    const scaleX = width ? next.w / width : 1;
    const scaleY = height ? next.h / height : 1;
    for (let i = 0; i < particleCount; i++) {
      px[i] *= scaleX;
      py[i] *= scaleY;
      pvx[i] *= scaleX;
      pvy[i] *= scaleY;
    }
    for (let i = 0; i < shellCount; i++) {
      sx[i] *= scaleX;
      sy[i] *= scaleY;
      svx[i] *= scaleX;
      svy[i] *= scaleY;
    }

    width = next.w;
    height = next.h;
    canvas.width = width;
    canvas.height = height;
    gravity = height * 1.05;
  };

  // Sparks, as parallel arrays: one allocation for the whole show.
  const px = new Float32Array(MAX_PARTICLES);
  const py = new Float32Array(MAX_PARTICLES);
  const pvx = new Float32Array(MAX_PARTICLES);
  const pvy = new Float32Array(MAX_PARTICLES);
  /** 1 at birth, 0 when dead. */
  const plife = new Float32Array(MAX_PARTICLES);
  const pdecay = new Float32Array(MAX_PARTICLES);
  const pcolor = new Uint8Array(MAX_PARTICLES);
  let particleCount = 0;

  const sx = new Float32Array(MAX_SHELLS);
  const sy = new Float32Array(MAX_SHELLS);
  const svx = new Float32Array(MAX_SHELLS);
  const svy = new Float32Array(MAX_SHELLS);
  const scolor = new Uint8Array(MAX_SHELLS);
  let shellCount = 0;

  // Scratch space for the draw order. Sparks are bucketed by (colour, alpha)
  // with a counting sort, which is one pass and no allocation.
  const bucketOf = new Uint16Array(MAX_PARTICLES);
  const bucketEnd = new Int32Array(BUCKETS);
  const bucketCount = new Int32Array(BUCKETS);
  const order = new Uint16Array(MAX_PARTICLES);

  resize();
  window.addEventListener('resize', resize);

  const launch = () => {
    if (shellCount >= MAX_SHELLS) return;
    const rise = rand(0.48, 0.72) * height;
    const i = shellCount++;
    // Kept away from the edges so the wider bursts stay on screen.
    sx[i] = rand(0.2, 0.8) * width;
    sy[i] = height + 10;
    svx[i] = rand(-0.05, 0.05) * width;
    svy[i] = -Math.sqrt(2 * gravity * rise);
    scolor[i] = Math.floor(Math.random() * COLORS.length);
  };

  const explode = (shellIndex: number, budget: number) => {
    const count = Math.min(Math.floor(rand(80, 120) * budget), MAX_PARTICLES - particleCount);
    const power = rand(0.5, 1.0) * height;
    const originX = sx[shellIndex];
    const originY = sy[shellIndex];
    const color = scolor[shellIndex];
    for (let n = 0; n < count; n++) {
      const angle = Math.random() * Math.PI * 2;
      // sqrt spreads the sparks evenly across the disc instead of clumping
      // them at the rim.
      const speed = power * Math.sqrt(Math.random()) * rand(0.75, 1.15);
      const i = particleCount++;
      px[i] = originX;
      py[i] = originY;
      pvx[i] = Math.cos(angle) * speed;
      pvy[i] = Math.sin(angle) * speed;
      plife[i] = 1;
      // Slightly slower burn, so the sparks reach the wider radius before
      // they fade out.
      pdecay[i] = rand(0.34, 0.9);
      pcolor[i] = color;
    }
  };

  const start = performance.now();
  let last = start;
  let nextLaunch = start;
  let frame = 0;
  let stopped = false;
  // Smoothed frame time, seeded at 60fps. Drives the spark budget below.
  let frameMs = 16.7;

  const cleanup = () => {
    stopped = true;
    cancelAnimationFrame(frame);
    window.removeEventListener('resize', resize);
  };

  const tick = (now: number) => {
    if (stopped) return;
    frame = requestAnimationFrame(tick);
    const raw = now - last;
    last = now;
    // Clamp so a dropped frame doesn't teleport everything.
    const dt = Math.min(raw / 1000, 0.05);
    const elapsed = now - start;
    const launching = elapsed < duration;

    // Trade sparks for frame rate, not the other way round: below ~45fps the
    // bursts thin out until the display can keep up again.
    frameMs += (Math.min(raw, 100) - frameMs) * 0.08;
    const budget = frameMs > 26 ? 0.5 : frameMs > 20 ? 0.75 : 1;

    if (launching && now >= nextLaunch) {
      launch();
      if (Math.random() < 0.3) launch();
      nextLaunch = now + rand(420, 800);
    }

    // Fade the previous frame rather than clearing it: this leaves the sparks'
    // trails while keeping the UI underneath untouched. The fade is derived
    // from dt so the trails stay the same length whatever the refresh rate —
    // a fixed per-frame fade makes them pulse whenever a frame runs long.
    const fade = Math.min(1, 1 - Math.pow(TRAIL_KEEP_PER_FRAME, dt * 60));
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = `rgba(0,0,0,${fade})`;
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'lighter';

    const shellSize = Math.max(2, Math.round(4 * renderScale));
    const shellOffset = shellSize >> 1;
    for (let i = shellCount - 1; i >= 0; i--) {
      svy[i] += gravity * dt;
      sx[i] += svx[i] * dt;
      sy[i] += svy[i] * dt;
      // Apex reached — burst.
      if (svy[i] >= 0) {
        explode(i, budget);
        const lastShell = --shellCount;
        if (i !== lastShell) {
          sx[i] = sx[lastShell];
          sy[i] = sy[lastShell];
          svx[i] = svx[lastShell];
          svy[i] = svy[lastShell];
          scolor[i] = scolor[lastShell];
        }
        continue;
      }
      ctx.fillStyle = COLORS[scolor[i]];
      ctx.fillRect((sx[i] - shellOffset) | 0, (sy[i] - shellOffset) | 0, shellSize, shellSize);
    }

    // Step the sparks, drop the dead ones and bucket the survivors, all in one
    // pass. Survivors are compacted towards the front of the arrays, so there
    // is no per-spark splice and no hole to skip over next frame.
    const drag = Math.pow(0.955, dt * 60);
    const fall = gravity * 0.34 * dt;
    const floor = height + 40;
    bucketCount.fill(0);
    let live = 0;
    for (let i = 0; i < particleCount; i++) {
      const life = plife[i] - pdecay[i] * dt;
      const vx = pvx[i] * drag;
      const vy = pvy[i] * drag + fall;
      const x = px[i] + vx * dt;
      const y = py[i] + vy * dt;
      if (life <= 0 || y > floor) continue;

      px[live] = x;
      py[live] = y;
      pvx[live] = vx;
      pvy[live] = vy;
      plife[live] = life;
      pdecay[live] = pdecay[i];
      const color = pcolor[i];
      pcolor[live] = color;

      let step = (life * ALPHA_STEPS) | 0;
      if (step >= ALPHA_STEPS) step = ALPHA_STEPS - 1;
      const bucket = color * ALPHA_STEPS + step;
      bucketOf[live] = bucket;
      bucketCount[bucket]++;
      live++;
    }
    particleCount = live;

    // Prefix sums, then place each spark in its bucket's slice of `order`.
    let offset = 0;
    for (let b = 0; b < BUCKETS; b++) {
      bucketEnd[b] = offset;
      offset += bucketCount[b];
    }
    for (let i = 0; i < live; i++) order[bucketEnd[bucketOf[i]]++] = i;

    // One fillStyle per bucket instead of one per spark: on a full screen of
    // sparks that is ~70 canvas state changes a frame rather than ~1000.
    const sizeBase = 4.5 * renderScale;
    const sizeSpan = 3.5 * renderScale;
    let cursor = 0;
    for (let b = 0; b < BUCKETS; b++) {
      const end = bucketEnd[b];
      if (end === cursor) continue;
      ctx.fillStyle = FILL_STYLES[b];
      for (let k = cursor; k < end; k++) {
        const i = order[k];
        // Integer rects skip the canvas antialiasing path, which is most of
        // the per-spark cost once the fills are batched.
        const size = Math.max(1, (sizeBase + plife[i] * sizeSpan) | 0);
        const half = size >> 1;
        ctx.fillRect((px[i] - half) | 0, (py[i] - half) | 0, size, size);
      }
      cursor = end;
    }

    ctx.globalCompositeOperation = 'source-over';

    if (!launching && !shellCount && !particleCount) {
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
      // Its own compositor layer, so repainting the canvas never drags the
      // board into a repaint with it (and vice versa).
      style={{ transform: 'translateZ(0)', willChange: 'transform', contain: 'layout paint style' }}
    />
  );
}
