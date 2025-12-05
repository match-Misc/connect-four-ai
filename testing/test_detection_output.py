#!/usr/bin/env python3
"""
Test detection output - continuously poll and display board state
"""
import json
import socket
import time

def bitmasks_to_grid(player1_mask, player2_mask):
    grid = [[0 for _ in range(7)] for _ in range(6)]
    for row in range(6):
        for col in range(7):
            bit_pos = (5 - row) * 7 + col
            if player1_mask & (1 << bit_pos):
                grid[row][col] = 1
            elif player2_mask & (1 << bit_pos):
                grid[row][col] = 2
    return grid

def connect_to_detection():
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect(("localhost", 65432))
        print("✓ Connected to detection server on port 65432\n")
        return client_socket
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return None

def main():
    client = connect_to_detection()
    if not client:
        return
    
    print("Polling detection server every 2 seconds...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            try:
                client.sendall(b"request")
                data = client.recv(1024)
                if data:
                    bitmasks = json.loads(data.decode("utf-8"))
                    p1 = bitmasks['player1']
                    p2 = bitmasks['player2']
                    
                    # Display board state
                    grid = bitmasks_to_grid(p1, p2)
                    symbols = ['.', '●', '○']  # empty, player1, player2
                    
                    print(f"Bitmasks: P1={p1}, P2={p2}")
                    for row in grid:
                        print('  ' + ' '.join(symbols[cell] for cell in row))
                    
                    # Count pieces
                    p1_count = sum(row.count(1) for row in grid)
                    p2_count = sum(row.count(2) for row in grid)
                    print(f"Pieces: P1={p1_count}, P2={p2_count}\n")
                else:
                    print("No data received")
                    break
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except socket.timeout:
                print("Socket timeout")
            except Exception as e:
                print(f"Error: {e}")
                break
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        client.close()

if __name__ == "__main__":
    main()
