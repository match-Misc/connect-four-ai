def validate(internal, merged, match_state, turn):
    missing_stones = False
    new_stones = []
    
    for r in range(6):
        for c in range(7):
            if internal[r][c] != 0 and merged[r][c] == 0:
                missing_stones = True
            elif internal[r][c] != 0 and merged[r][c] != internal[r][c]:
                missing_stones = True
            elif internal[r][c] == 0 and merged[r][c] != 0:
                new_stones.append([r, c, merged[r][c]])
                
    if missing_stones:
        return "Stone(s) removed or altered unexpectedly!", []
        
    if len(new_stones) > 1:
        return "Too many stones inserted at once! Please remove the extra stones.", [[r, c] for r, c, p in new_stones]
        
    if len(new_stones) == 1:
        r, c, p = new_stones[0]
        if match_state != "in_game":
            return "Game is not active. Please Start/Reset the game.", [[r, c]]
        if (turn == "human" and p != 1) or (turn == "robot" and p != 2):
            return f"Wrong token inserted! Expected Player {1 if turn == 'human' else 2} ({turn}).", [[r, c]]
            
        return None, [] # valid
        
    return None, [] # unchanged

print("Loaded")
