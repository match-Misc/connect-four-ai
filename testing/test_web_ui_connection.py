#!/usr/bin/env python3
"""
Test script to verify web-ui can connect to detection socket server
"""
import json
import socket
import time

def test_connection():
    print("Testing connection to detection socket server...")
    
    try:
        # Try to connect to detection server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5)
        client_socket.connect(("localhost", 65432))
        print("✓ Successfully connected to detection server on port 65432")
        
        # Send request
        client_socket.sendall(b"request")
        print("✓ Sent request to detection server")
        
        # Receive response
        data = client_socket.recv(1024)
        if data:
            bitmasks = json.loads(data.decode("utf-8"))
            print(f"✓ Received bitmasks from detection server:")
            print(f"  - player1: {bitmasks['player1']}")
            print(f"  - player2: {bitmasks['player2']}")
            
            # Convert to grid for visualization
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
            
            grid = bitmasks_to_grid(bitmasks['player1'], bitmasks['player2'])
            print("\nBoard state:")
            symbols = ['.', '●', '○']  # empty, player1, player2
            for row in grid:
                print('  ' + ' '.join(symbols[cell] for cell in row))
            
            print("\n✓ Connection test PASSED!")
            return True
        else:
            print("✗ No data received from detection server")
            return False
            
    except socket.timeout:
        print("✗ Connection timeout - is detection.py running?")
        return False
    except ConnectionRefusedError:
        print("✗ Connection refused - detection.py socket server not running")
        print("  Start detection with: python robot-game/detection.py")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON response: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        try:
            client_socket.close()
        except:
            pass

if __name__ == "__main__":
    print("=" * 60)
    print("Web-UI to Detection Connection Test")
    print("=" * 60)
    print()
    
    success = test_connection()
    
    print()
    if success:
        print("The web-ui should be able to connect to detection successfully!")
    else:
        print("Please ensure detection.py is running before starting web-ui")
        print("Run: python robot-game/detection.py")
