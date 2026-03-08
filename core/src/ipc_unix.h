#pragma once

#include "ipc_server.h"

#ifndef _WIN32

class UnixSocketServer : public IpcServer {
public:
  UnixSocketServer();
  ~UnixSocketServer() override;

  bool pollCommand(uint8_t &cmdByte) override;

private:
  int sockFd;
  int clientFd;
};

#endif
