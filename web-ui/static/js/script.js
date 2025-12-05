// Connect to SocketIO
const socket = io();

// Local grid state for testing
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

// Listen for board updates
socket.on('board_update', function(data) {
    // Update local grid with server data
    localGrid = data.grid.map(row => [...row]);
    updateBoard(localGrid);
});

// Handle column clicks for testing
document.addEventListener('DOMContentLoaded', function() {
    const overlays = document.querySelectorAll('.column-overlay');
    overlays.forEach(overlay => {
        overlay.addEventListener('click', function() {
            const col = parseInt(this.dataset.col);
            dropChip(col);
        });
    });
    
    // Initialize player indicators
    updatePlayerIndicators(currentPlayer, currentPlayer === 1 ? 2 : 1);
});

function dropChip(col) {
    // Find the lowest empty row in the column
    for (let row = 5; row >= 0; row--) {
        if (localGrid[row][col] === 0) {
            animateChipDrop(col, row);
            localGrid[row][col] = currentPlayer;
            const previousPlayer = currentPlayer;
            currentPlayer = currentPlayer === 1 ? 2 : 1; // Switch player
            updatePlayerIndicators(currentPlayer, previousPlayer);
            // Update after animation
            setTimeout(() => updateBoard(localGrid), 600);
            break;
        }
    }
}

function animateChipDrop(col, targetRow) {
    const chip = document.createElement('div');
    chip.className = `falling-chip player${currentPlayer}`;
    chip.style.left = `${25 + col * 60}px`; // Position at top of column
    chip.style.top = '0px';
    document.getElementById('game-board').appendChild(chip);

    // Animate falling
    const targetTop = 20 + targetRow * 60; // 20px padding + row * (50px cell + 10px margin)
    chip.style.transition = 'top 0.6s ease-in';
    setTimeout(() => {
        chip.style.top = `${targetTop}px`;
    }, 10);

    // Remove after animation
    setTimeout(() => {
        chip.remove();
    }, 600);
}

function updateBoard(grid) {
    for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 7; col++) {
            const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
            const player = grid[row][col];
            cell.className = 'cell'; // Reset classes
            if (player === 1) {
                cell.classList.add('player1');
            } else if (player === 2) {
                cell.classList.add('player2');
            }
        }
    }
}

console.log('Flask web UI loaded');