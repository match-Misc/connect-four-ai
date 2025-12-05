// Connect to SocketIO
const socket = io();

// Local grid state - will be updated from detection server
let localGrid = Array(6).fill().map(() => Array(7).fill(0));
let currentPlayer = 1;

function updatePlayerIndicators(activePlayer, inactivePlayer) {
    const activeIndicator = document.getElementById(`player${activePlayer}-indicator`);
    const inactiveIndicator = document.getElementById(`player${inactivePlayer}-indicator`);
    
    // Reset both indicators to ensure clean state
    document.getElementById('player1-indicator').classList.remove('active', 'inactive');
    document.getElementById('player2-indicator').classList.remove('active', 'inactive');
    
    // Set active player
    activeIndicator.classList.add('active');
    // Set inactive player  
    inactiveIndicator.classList.add('inactive');
}

// Listen for board updates from detection system
socket.on('board_update', function(data) {
    console.log('Received board update:', data);
    console.log('Grid data type:', typeof data.grid, 'Length:', data.grid ? data.grid.length : 'N/A');
    // Update local grid with server data
    localGrid = data.grid.map(row => [...row]);
    console.log('Local grid after update:', localGrid);
    updateBoard(localGrid);
    
    // Count pieces to determine current player
    let player1Count = 0;
    let player2Count = 0;
    for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 7; col++) {
            if (localGrid[row][col] === 1) player1Count++;
            if (localGrid[row][col] === 2) player2Count++;
        }
    }
    
    // Determine whose turn based on piece count
    if (player1Count <= player2Count) {
        currentPlayer = 1;
    } else {
        currentPlayer = 2;
    }
    updatePlayerIndicators(currentPlayer, currentPlayer === 1 ? 2 : 1);
});

// Socket connection status
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
});

// Handle column clicks - DISABLED FOR DETECTION MODE
document.addEventListener('DOMContentLoaded', function() {
    // Disable manual play by commenting out click handlers
    /*
    const overlays = document.querySelectorAll('.column-overlay');
    overlays.forEach(overlay => {
        overlay.addEventListener('click', function() {
            const col = parseInt(this.dataset.col);
            dropChip(col);
        });
    });
    */
    
    // Initialize player indicators
    updatePlayerIndicators(currentPlayer, currentPlayer === 1 ? 2 : 1);
    
    console.log('Web UI initialized in detection mode (manual play disabled)');
});

function updateBoard(grid) {
    console.log('updateBoard called with:', grid);
    let piecesFound = 0;
    for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 7; col++) {
            const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
            if (!cell) {
                console.error(`Cell not found at row ${row}, col ${col}`);
                continue;
            }
            const player = grid[row][col];
            if (player !== 0) {
                piecesFound++;
                console.log(`Piece found at (${row}, ${col}): player ${player}`);
            }
            cell.className = 'cell'; // Reset classes
            if (player === 1) {
                cell.classList.add('player1');
                console.log(`Added player1 class to cell (${row}, ${col})`);
            } else if (player === 2) {
                cell.classList.add('player2');
                console.log(`Added player2 class to cell (${row}, ${col})`);
            }
        }
    }
    console.log(`Board updated with grid. Total pieces found: ${piecesFound}`, grid);
}

console.log('Connect Four Web UI loaded - Detection Mode Active');