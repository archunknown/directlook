#include "ipc_unix.h"

#ifndef _WIN32
#include "protocol.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

UnixSocketServer::UnixSocketServer() : sockFd(-1), clientFd(-1) {
  sockFd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);

  if (sockFd < 0) {
    throw std::runtime_error(
        "Falla crítica: Imposible instanciar Socket UNIX IPC.");
  }

  unlink(DIRECTLOOK_SOCK_PATH);

  struct sockaddr_un addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, DIRECTLOOK_SOCK_PATH, sizeof(addr.sun_path) - 1);

  if (bind(sockFd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
      0) {
    throw std::runtime_error(
        "Falla crítica: Imposible vincular Socket UNIX IPC.");
  }
  if (listen(sockFd, 1) < 0) {
    throw std::runtime_error(
        "Falla crítica: Imposible escuchar Módulos en Socket UNIX IPC.");
  }
  std::cout << "[IPC] Socket UNIX creado: " << DIRECTLOOK_SOCK_PATH
            << std::endl;
}

UnixSocketServer::~UnixSocketServer() {
  if (clientFd >= 0)
    close(clientFd);
  if (sockFd >= 0) {
    close(sockFd);
    unlink(DIRECTLOOK_SOCK_PATH);
    std::cout << "[IPC] Socket UNIX cerrado y desvinculado." << std::endl;
  }
}

bool UnixSocketServer::pollCommand(uint8_t &cmdByte) {
  if (sockFd < 0)
    return false;

  if (clientFd < 0) {
    clientFd = accept(sockFd, NULL, NULL);
    if (clientFd >= 0) {
      int flags = fcntl(clientFd, F_GETFL, 0);
      fcntl(clientFd, F_SETFL, flags | O_NONBLOCK);
    }
  }

  if (clientFd >= 0) {
    ssize_t n = recv(clientFd, &cmdByte, 1, MSG_DONTWAIT);
    if (n == 1) {
      return true;
    } else if (n == 0) {
      // Client closed connection
      close(clientFd);
      clientFd = -1;
    }
  }

  return false;
}

#endif
