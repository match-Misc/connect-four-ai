import { createRoot } from 'react-dom/client';
import { Draw } from './components/Draw';
import { RobotWins } from './components/RobotWins';
import './index.css';

// Flags come in through the hash, comma-separated: `#draw`, `#dark`, `#draw,dark`.
const flags = new Set(location.hash.slice(1).split(','));

if (flags.has('dark')) document.documentElement.classList.add('dark');

const Overlay = flags.has('draw') ? Draw : RobotWins;

createRoot(document.getElementById('root')!).render(
  <div className="h-screen w-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
    <div className="w-[70vw] h-[70vh] rounded-3xl bg-blue-600" />
    <Overlay active duration={60000} />
  </div>,
);
