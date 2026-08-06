import serial
import time
import threading

def nfc_worker(callback, port='/dev/ttyUSB0', baud=9600):
    """
    Background worker that continuously reads from the NFC serial port.
    """
    while True:
        ser = None
        try:
            # Reconnect loop
            ser = serial.Serial(port, baud, timeout=0.5)
            print(f"[NFC] Connected to {port} at {baud} baud.")
            
            while True:
                try:
                    if ser.in_waiting > 0:
                        line = ser.readline()
                        if line:
                            try:
                                decoded = line.decode('utf-8', errors='ignore').strip()
                                if decoded:
                                    print(f"[NFC] Tag scanned: {decoded}")
                                    callback(decoded)
                            except Exception as e:
                                print(f"[NFC] Decode error: {e}")
                except OSError:
                    # Device disconnected or error
                    print(f"[NFC] Device disconnected, reconnecting...")
                    break
                    
                time.sleep(0.01)
                
        except Exception as e:
            # Reader might be unplugged, wait before retrying
            time.sleep(2)
        finally:
            if ser is not None:
                try:
                    ser.close()
                except:
                    pass

def start_nfc_reader(callback):
    """
    Starts the NFC reader in a daemon thread.
    Returns the thread object.
    """
    t = threading.Thread(target=nfc_worker, args=(callback,), daemon=True)
    t.start()
    return t
