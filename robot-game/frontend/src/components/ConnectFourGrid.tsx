import { cn } from '../utils';

interface ConnectFourGridProps {
  board: number[][];
  onColumnClick: (col: number) => void;
  disabled: boolean;
  invalidStones?: number[][];
  robotTargetCol?: number | null;
}

export function ConnectFourGrid({ board, onColumnClick, disabled, invalidStones = [], robotTargetCol = null }: ConnectFourGridProps) {
  return (
    <div className="relative bg-gray-700 p-2 sm:p-4 rounded-2xl shadow-xl border-4 border-gray-800 w-full h-full">
      <div className="grid grid-cols-7 grid-rows-6 gap-2 sm:gap-3 w-full h-full">
        {board.map((row, rowIndex) => 
          row.map((cell, colIndex) => {
            const isInvalid = invalidStones.some(([r, c]) => r === rowIndex && c === colIndex);
            
            return (
            <div 
              key={`${rowIndex}-${colIndex}`} 
              className={cn("w-full h-full rounded-full flex items-center justify-center cursor-pointer transition-transform hover:scale-[1.03]",
                isInvalid ? "bg-red-900/50 animate-pulse" : "bg-gray-800"
              )}
              onClick={() => {
                if (!disabled) {
                  onColumnClick(colIndex);
                }
              }}
            >
              <div className={cn(
                "w-[85%] h-[85%] rounded-full shadow-[inset_0_2px_4px_rgba(0,0,0,0.6)] transition-all duration-300",
                cell === 0 ? "bg-white border-2 border-gray-200" : 
                cell === 1 ? "bg-brand-green border-2 border-[#8C9E17]" : 
                "bg-gray-900 border-2 border-black",
                isInvalid && "ring-4 ring-red-500 shadow-[0_0_15px_rgba(239,68,68,0.8)]"
              )} />
            </div>
          )})
        )}
      </div>
      
      {robotTargetCol !== null && (
        <div 
          className="absolute top-0 w-full h-full pointer-events-none"
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(7, minmax(0, 1fr))',
            gap: '0.5rem',
            padding: '1rem'
          }}
        >
          {Array(7).fill(0).map((_, col) => (
            <div key={col} className="w-full h-full flex justify-center">
              {col === robotTargetCol && (
                <div className="w-8 h-8 sm:w-12 sm:h-12 mt-2 bg-gradient-to-b from-purple-400 to-purple-600 rounded-full flex items-center justify-center animate-bounce shadow-[0_0_20px_rgba(168,85,247,0.8)] border-2 border-white text-white font-black text-xl z-50 shadow-purple-500/50">
                  ↓
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
