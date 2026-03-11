#pragma once

#include <cstdint>

class IpcServer {
public:
  virtual ~IpcServer() = default;

  // Checks for an incoming byte asynchronously.
  // Returns true if a command byte was successfully read, false otherwise.
  virtual bool pollCommand(uint8_t &cmdByte) = 0;

  // Sends a telemetry byte to the connected client asynchronously.
  // Fails silently if no client is connected or buffer is full.
  virtual void sendTelemetry(uint8_t code) = 0;
};
