#pragma once

#include <cstdint>

class IpcServer {
public:
  virtual ~IpcServer() = default;

  // Checks for an incoming byte asynchronously.
  // Returns true if a command byte was successfully read, false otherwise.
  virtual bool pollCommand(uint8_t &cmdByte) = 0;
};
