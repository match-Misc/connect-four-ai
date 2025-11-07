#!/usr/bin/env python3
"""
Robot Simulator for Connect Four

This script simulates a UR10e robot by listening for TCP messages containing
column numbers where the robot should place tiles. It prints the received
column numbers to the console for testing purposes.
"""

import json
import socket
import sys
import time


class RobotSimulator:
    """Simulates a robot that receives column numbers via TCP."""

    def __init__(self, host: str = "localhost", port: int = 30002):
        self.host = host
        self.port = port
        self.server_socket = None

    def start_server(self):
        """Start the TCP server to listen for robot commands."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Robot simulator listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to start robot simulator: {e}")
            return False

    def run(self):
        """Main loop to accept connections and process commands."""
        if not self.start_server():
            return

        try:
            while True:
                print("Waiting for connection...")
                client_socket, client_address = self.server_socket.accept()
                print(f"Connected to {client_address}")

                try:
                    while True:
                        data = client_socket.recv(1024)
                        if not data:
                            break

                        try:
                            # Expecting a simple integer as string
                            column = int(data.decode("utf-8").strip())
                            print(
                                f"Robot received command: Place tile in column {column + 1} (0-indexed: {column})"
                            )

                            # Simulate robot action (just print for now)
                            print(
                                f"Simulating robot placing tile in column {column + 1}..."
                            )

                            # Send acknowledgment
                            client_socket.sendall(b"OK")

                        except ValueError as e:
                            print(f"Invalid data received: {data} - {e}")
                            client_socket.sendall(b"ERROR")

                except Exception as e:
                    print(f"Error handling client: {e}")
                finally:
                    client_socket.close()
                    print("Client disconnected")

        except KeyboardInterrupt:
            print("Robot simulator shutting down")
        finally:
            if self.server_socket:
                self.server_socket.close()


def main():
    """Main entry point."""
    # Load configuration
    try:
        with open("robotconfig.json", "r") as f:
            config = json.load(f)
        host = config.get("robot_ip", "localhost")
        port = config.get("robot_port", 30002)
    except Exception as e:
        print(f"Could not load robotconfig.json, using defaults: {e}")
        host = "localhost"
        port = 30002

    simulator = RobotSimulator(host, port)
    simulator.run()


if __name__ == "__main__":
    main()
