#pragma once
#include <cstdint>

// Comandos IPC (protocolo binario de 1 byte)
constexpr uint8_t DIRECTLOOK_CMD_DISABLE       = 0x00;
constexpr uint8_t DIRECTLOOK_CMD_ENABLE         = 0x01;
constexpr uint8_t DIRECTLOOK_CMD_THERMAL_ALARM  = 0x03;

// Canales de transporte por plataforma
#ifdef _WIN32
constexpr const char* DIRECTLOOK_PIPE_NAME = "\\\\.\\pipe\\directlook_pipe";
#else
constexpr const char* DIRECTLOOK_SOCK_PATH = "/tmp/directlook.sock";
#endif