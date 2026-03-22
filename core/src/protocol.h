#pragma once
#include <cstdint>

constexpr uint8_t DIRECTLOOK_CMD_DISABLE = 0x00;
constexpr uint8_t DIRECTLOOK_CMD_ENABLE = 0x01;
constexpr uint8_t DIRECTLOOK_CMD_THERMAL_ALARM = 0xFF;
constexpr const char *DIRECTLOOK_SOCK_PATH = "/tmp/directlook.sock";