"""
DirectLook — Test IPC (Windows Named Pipe)
Protocolo binario de 1 byte: 0x00 = desactivar, 0x01 = activar.
Uso: python test_ipc.py        (envia 0x00 = desactivar)
      python test_ipc.py on     (envia 0x01 = activar)
"""
import sys
import time
import os
import socket

byte = b'\x00' if len(sys.argv) < 2 or sys.argv[1] != 'on' else b'\x01'
label = "ENABLE (0x01)" if byte == b'\x01' else "DISABLE (0x00)"

if os.name == 'nt':
    # Windows: Named Pipe
    PIPE_PATH = r'\\.\pipe\directlook_pipe'
    try:
        pipe = open(PIPE_PATH, 'r+b', buffering=0)
        time.sleep(0.1)
        pipe.write(byte)
        pipe.flush()
        time.sleep(0.1)
        pipe.close()
        print(f"[OK] Comando {label} enviado al daemon (Windows).")
    except FileNotFoundError:
        print("[ERROR] El daemon no esta corriendo (pipe no encontrado).")
    except Exception as e:
        print(f"[ERROR] {e}")
else:
    # Linux: Unix Domain Socket
    SOCK_PATH = '/tmp/directlook.sock'
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(SOCK_PATH)
        client.send(byte)
        client.close()
        print(f"[OK] Comando {label} enviado al daemon (Linux).")
    except FileNotFoundError:
        print("[ERROR] El daemon no esta corriendo (socket no encontrado).")
    except Exception as e:
        print(f"[ERROR] {e}")
