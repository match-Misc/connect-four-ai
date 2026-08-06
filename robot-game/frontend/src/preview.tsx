import { createRoot } from 'react-dom/client';
import { RobotWins } from './components/RobotWins';
import './index.css';

if (location.hash === '#dark') document.documentElement.classList.add('dark');

createRoot(document.getElementById('root')!).render(
  <div className="h-screen w-screen bg-gray-50 dark:bg-gray-950 flex items-center justify-center">
    <div className="w-[70vw] h-[70vh] rounded-3xl bg-blue-600" />
    <RobotWins active duration={60000} />
  </div>,
);
