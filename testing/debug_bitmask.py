#!/usr/bin/env python3
"""Debug script to test bitmask conversion"""

def bitmasks_to_grid(player1_mask, player2_mask):
    grid = [[0 for _ in range(7)] for _ in range(6)]
    for row in range(6):
        for col in range(7):
            bit_pos = (5 - row) * 7 + col
            if player1_mask & (1 << bit_pos):
                grid[row][col] = 1  # Player 1
            elif player2_mask & (1 << bit_pos):
                grid[row][col] = 2  # Player 2
    return grid

# Test with a simple pattern: one piece in bottom-left (bit 0)
player1_mask = 1  # Bit 0 set
player2_mask = 0

grid = bitmasks_to_grid(player1_mask, player2_mask)
print("Test 1: Player 1 piece at bit 0 (bottom-left)")
print("Player1 mask:", player1_mask)
print("Grid:")
for i, row in enumerate(grid):
    print(f"  Row {i}: {row}")
print()

# Test with piece in top-left (bit 35)
player1_mask = 1 << 35
player2_mask = 0

grid = bitmasks_to_grid(player1_mask, player2_mask)
print("Test 2: Player 1 piece at bit 35 (top-left)")
print("Player1 mask:", player1_mask)
print("Grid:")
for i, row in enumerate(grid):
    print(f"  Row {i}: {row}")
print()

# Test with pieces in bottom row (bits 0-6)
player1_mask = 0b1010101  # Bits 0, 2, 4, 6
player2_mask = 0b0101010  # Bits 1, 3, 5

grid = bitmasks_to_grid(player1_mask, player2_mask)
print("Test 3: Alternating pieces in bottom row")
print(f"Player1 mask: {player1_mask:b}")
print(f"Player2 mask: {player2_mask:b}")
print("Grid:")
for i, row in enumerate(grid):
    print(f"  Row {i}: {row}")
