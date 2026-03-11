#include "ipc_windows.h"

#ifdef _WIN32
#include "protocol.h"
#include <iostream>
#include <stdexcept>

WindowsNamedPipeServer::WindowsNamedPipeServer()
    : hPipe(INVALID_HANDLE_VALUE), pipeConnected(false), connectPending(false),
      readPending(false) {
  std::memset(&olConnect, 0, sizeof(olConnect));
  std::memset(&olRead, 0, sizeof(olRead));
  std::memset(&olWrite, 0, sizeof(olWrite));

  hPipe = CreateNamedPipeA(DIRECTLOOK_PIPE_NAME,
                           PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                           PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                           1, // max instances
                           1, // out buffer
                           1, // in buffer
                           0, // timeout
                           NULL);

  olConnect.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
  olRead.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
  olWrite.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

  if (hPipe == INVALID_HANDLE_VALUE) {
    throw std::runtime_error(
        "Falla crítica: Imposible instanciar Named Pipe IPC.");
  }

  std::cout << "[IPC] Named Pipe creado: " << DIRECTLOOK_PIPE_NAME << std::endl;
  ConnectNamedPipe(hPipe, &olConnect);
  DWORD err = GetLastError();
  if (err == ERROR_IO_PENDING) {
    connectPending = true;
  } else if (err == ERROR_PIPE_CONNECTED) {
    pipeConnected = true;
    std::cout << "[IPC] Cliente conectado." << std::endl;
  }
}

WindowsNamedPipeServer::~WindowsNamedPipeServer() {
  if (hPipe != INVALID_HANDLE_VALUE) {
    if (readPending)
      CancelIo(hPipe);
    if (pipeConnected)
      DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
    if (olConnect.hEvent)
      CloseHandle(olConnect.hEvent);
    if (olRead.hEvent)
      CloseHandle(olRead.hEvent);
    if (olWrite.hEvent)
      CloseHandle(olWrite.hEvent);
    std::cout << "[IPC] Named Pipe cerrado." << std::endl;
  }
}

bool WindowsNamedPipeServer::pollCommand(uint8_t &cmdByte) {
  if (hPipe == INVALID_HANDLE_VALUE)
    return false;

  if (!pipeConnected && connectPending) {
    DWORD dummy;
    if (GetOverlappedResult(hPipe, &olConnect, &dummy, FALSE)) {
      pipeConnected = true;
      connectPending = false;
    }
  }

  if (pipeConnected) {
    if (!readPending) {
      DWORD bytesRead = 0;
      if (ReadFile(hPipe, &cmdByte, 1, &bytesRead, &olRead)) {
        return true;
      } else if (GetLastError() == ERROR_IO_PENDING) {
        readPending = true;
      } else {
        DisconnectNamedPipe(hPipe);
        pipeConnected = false;
        readPending = false;
        ResetEvent(olConnect.hEvent);
        ConnectNamedPipe(hPipe, &olConnect);
        connectPending = (GetLastError() == ERROR_IO_PENDING);
      }
    } else {
      DWORD bytesRead = 0;
      if (GetOverlappedResult(hPipe, &olRead, &bytesRead, FALSE)) {
        readPending = false;
        if (bytesRead == 1) {
          return true;
        }
      } else if (GetLastError() != ERROR_IO_INCOMPLETE) {
        DisconnectNamedPipe(hPipe);
        pipeConnected = false;
        readPending = false;
        ResetEvent(olConnect.hEvent);
        ConnectNamedPipe(hPipe, &olConnect);
        connectPending = (GetLastError() == ERROR_IO_PENDING);
      }
    }
  }
  return false;
}

void WindowsNamedPipeServer::sendTelemetry(uint8_t code) {
  if (hPipe == INVALID_HANDLE_VALUE || !pipeConnected) return;
  DWORD bytesWritten = 0;
  WriteFile(hPipe, &code, 1, &bytesWritten, &olWrite);
}

#endif
