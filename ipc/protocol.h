#pragma once
// =============================================================================
// DirectLook — Protocolo IPC (Capa 1: Telemetría)
// Definiciones compartidas entre Daemon Core y Client GUI
// =============================================================================

// --- Canales de transporte por plataforma ---
#ifdef _WIN32
#define DIRECTLOOK_PIPE_NAME "\\\\.\\pipe\\directlook_pipe"
#else
#define DIRECTLOOK_SOCK_PATH "/tmp/directlook.sock"
#endif

// --- Protocolo binario de 1 byte (zero-overhead) ---
#define DIRECTLOOK_CMD_DISABLE 0x00
#define DIRECTLOOK_CMD_ENABLE 0x01
#define DIRECTLOOK_CMD_THERMAL_ALARM 0x03
