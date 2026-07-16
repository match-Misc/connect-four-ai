#!/usr/bin/env python3
"""
Simple test client to connect to the detection socket server
and receive bitmask updates.
"""

import json
import socket
import time


def test_socket_client():
    """Test client to connect to socket server and print bitmasks"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(("localhost", 65432))
        print("Connected to socket server")

        while True:
            # Send a request (any data will trigger response)
            client_socket.sendall(b"request")

            # Receive response
            data = client_socket.recv(1024)
            if not data:
                break

            # Parse JSON response
            try:
                bitmasks = json.loads(data.decode("utf-8"))
                print(f"Player 1: {bitmasks['player1']}")
                print(f"Player 2: {bitmasks['player2']}")
                print("-" * 40)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")

            time.sleep(1)  # Wait 1 second before next request

    except KeyboardInterrupt:
        print("Test client stopped")
    except ConnectionRefusedError:
        print("Connection refused - is the detection server running?")
    finally:
        client_socket.close()


if __name__ == "__main__":
    test_socket_client()
