#pragma once

#include "ipc_server.h"

#ifdef _WIN32
#define NOMINMAX
#include <string>
#include <windows.h>

class WindowsNamedPipeServer : public IpcServer {
public:
  WindowsNamedPipeServer();
  ~WindowsNamedPipeServer() override;

  bool pollCommand(uint8_t &cmdByte) override;
  void sendTelemetry(uint8_t code) override;

private:
  HANDLE hPipe;
  OVERLAPPED olConnect;
  OVERLAPPED olRead;
  OVERLAPPED olWrite;
  bool pipeConnected;
  bool connectPending;
  bool readPending;
};

#endif
