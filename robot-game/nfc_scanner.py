#!/usr/bin/env python3
"""
NFC Scanner Script for Connect Four Game

This script connects to an ESP32/D1 Mini via serial (COM11) and reads NFC tag IDs.
It parses the NFC_ID from the serial output and returns it for API integration.

Expected serial output format from ESP: "NFC_ID:XXXXXXXX"
"""

import re
import time

import serial


class NFCScanner:
    def __init__(self, port="COM11", baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None

    def connect(self):
        """Connect to the ESP32 serial port."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )
            print(f"Connected to NFC scanner on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from the serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from NFC scanner")

    def read_nfc_id(self, timeout=10):
        """
        Read NFC ID from serial output.

        Args:
            timeout (int): Maximum time to wait for NFC scan in seconds

        Returns:
            str or None: NFC ID if found, None if timeout or error
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial connection not open")
            return None

        start_time = time.time()
        nfc_pattern = re.compile(r"NFC_ID:([A-Fa-f0-9]+)")

        print("Waiting for NFC tag scan...")

        while time.time() - start_time < timeout:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode("utf-8").strip()
                    print(f"Received: {line}")

                    match = nfc_pattern.search(line)
                    if match:
                        nfc_id = match.group(1).upper()
                        print(f"NFC tag detected: {nfc_id}")
                        return nfc_id

            except serial.SerialException as e:
                print(f"Serial read error: {e}")
                return None

            time.sleep(0.1)

        print("Timeout: No NFC tag detected")
        return None


def scan_nfc_tag(port="COM11", timeout=10):
    """
    Convenience function to scan for an NFC tag.

    Args:
        port (str): Serial port (default: COM11)
        timeout (int): Timeout in seconds

    Returns:
        str or None: NFC ID if successful, None otherwise
    """
    scanner = NFCScanner(port=port)
    if not scanner.connect():
        return None

    try:
        return scanner.read_nfc_id(timeout=timeout)
    finally:
        scanner.disconnect()


if __name__ == "__main__":
    # Test the scanner
    nfc_id = scan_nfc_tag()
    if nfc_id:
        print(f"Scanned NFC ID: {nfc_id}")
    else:
        print("Failed to scan NFC tag")
