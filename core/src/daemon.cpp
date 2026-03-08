// =============================================================================
// DirectLook — Daemon Core (Multiplataforma: Windows + Linux)
// Sprint 2: Servicio Continuo · Canal IPC · Protocolo de Telemetría
// =============================================================================

// --- Includes comunes (agnósticos de plataforma) ---
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ipc_server.h"
#include "protocol.h"
#include "vision_pipeline.h"

// --- Includes condicionales de plataforma ---

#ifdef _WIN32
#define NOMINMAX
#include "ipc_windows.h"
#include "video_sink_windows.h"
#include <psapi.h>
#include <windows.h>
#else

#include "ipc_unix.h"
#include "video_sink_unix.h"
#include <fcntl.h>
#include <fstream>
#include <sys/ioctl.h>
#include <unistd.h>

#endif

// =============================================================================
// Variable atómica global de control de ejecución
// =============================================================================
std::atomic<bool> keepRunning{true};

void signalHandler(int signum) {
  keepRunning.store(false);
  // write() es async-signal-safe; cout no lo es, pero es aceptable aquí
  const char *msg = "\n[SIGNAL] Señal interceptada. Iniciando shutdown...\n";
#ifdef _WIN32
  DWORD written;
  WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), msg,
            static_cast<DWORD>(strlen(msg)), &written, NULL);
#else
  (void)write(STDOUT_FILENO, msg, strlen(msg));
#endif
  (void)signum;
}

// =============================================================================
// Utilidades Multiplataforma
// =============================================================================
std::string getExecutableDir() {
#ifdef _WIN32
  char path[MAX_PATH];
  GetModuleFileNameA(NULL, path, MAX_PATH);
  std::string fullPath(path);
  size_t pos = fullPath.find_last_of("\\/");
  return (pos == std::string::npos) ? "" : fullPath.substr(0, pos);
#else
  char path[1024];
  ssize_t count = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (count <= 0)
    return "";
  path[count] = '\0';
  std::string fullPath(path);
  size_t pos = fullPath.find_last_of('/');
  return (pos == std::string::npos) ? "" : fullPath.substr(0, pos);
#endif
}

std::string resolveModelPath(const std::string &modelName) {
  std::string exeDir = getExecutableDir();
  // Posibles ubicaciones relativas al ejecutable:
  // 1. Mismo directorio (producción flat)
  // 2. Un nivel arriba (típico Linux: build/core -> build/modelos)
  // 3. Dos niveles arriba (típico MSVC: build/core/Release -> build/modelos)
  std::vector<std::string> searchPaths = {
      exeDir + "/modelos/" + modelName, exeDir + "/../modelos/" + modelName,
      exeDir + "/../../modelos/" + modelName};

  for (const auto &path : searchPaths) {
    // Verificación rápida de existencia de archivo usando fopen
    FILE *f = fopen(path.c_str(), "rb");
    if (f) {
      fclose(f);
      return path;
    }
  }
  return exeDir + "/modelos/" + modelName; // Fallback
}

// =============================================================================
// Constantes y variables arquitectónicas
// =============================================================================
static constexpr size_t MEMORY_LIMIT_BYTES = 125 * 1024 * 1024; // 125 MB
std::string MODEL_PATH;
std::string FACE_MODEL_PATH;

// =============================================================================
// [FASE 1] Monitor de Memoria Multiplataforma
// =============================================================================
size_t getProcessMemory() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX pmcEx;
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmcEx),
                           sizeof(pmcEx))) {
    return static_cast<size_t>(pmcEx.PrivateUsage);
  }
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      size_t kb = 0;
      std::sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb * 1024;
    }
  }
  return 0;
#endif
}

// =============================================================================
// Utilidad: Procesar comando IPC (protocolo binario de 1 byte)
// =============================================================================
bool processIpcCommand(uint8_t byte, bool &effectEnabled) {
  if (byte == DIRECTLOOK_CMD_DISABLE) {
    effectEnabled = false;
    std::cout << "[IPC] Efecto DESACTIVADO (0x00)" << std::endl;
    return true;
  } else if (byte == DIRECTLOOK_CMD_ENABLE) {
    effectEnabled = true;
    std::cout << "[IPC] Efecto ACTIVADO (0x01)" << std::endl;
    return true;
  }
  return false;
}

// =============================================================================
// Arquitecturas de plataforma (main separados)
// =============================================================================

#ifdef _WIN32

// =====================================================================
// ARQUITECTURA WINDOWS (DirectShow / MSMF + Named Pipe IPC)
// =====================================================================

int main() {
  // --- Registro de señales ---
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::cout << "=== DirectLook Daemon [Windows] ===" << std::endl;
  std::cout << "[DAEMON] Modo servicio continuo activo." << std::endl;

  // Inicializar rutas dinámicas con resolución heurística
  MODEL_PATH = resolveModelPath("pfld.onnx");
  FACE_MODEL_PATH = resolveModelPath("version-slim-320_simplified.onnx");

  // Variables para limpieza garantizada (RAII-like handling)
  std::unique_ptr<IpcServer> ipcServer =
      std::make_unique<WindowsNamedPipeServer>();
  std::unique_ptr<VideoSink> videoSink =
      std::make_unique<WindowsVirtualCamSink>();

  cv::VideoCapture cap;
  size_t frames_processed = 0;
  double total_latency = 0.0;

  try {

    // Reporte de memoria pre-inicialización
    size_t memInicio = getProcessMemory();
    std::cout << "[MEMORIA] Inicio del proceso: "
              << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

    // -----------------------------------------------------------------
    // Instanciación del Motor ONNX Runtime (CPU)
    // -----------------------------------------------------------------
    VisionPipeline vision(FACE_MODEL_PATH, MODEL_PATH);

    // -----------------------------------------------------------------
    // Captura de video
    // -----------------------------------------------------------------
    bool effectEnabled = true;
    uint8_t asyncCmdByte = 0;

    // -----------------------------------------------------------------
    // Bucle principal perpetuo
    // -----------------------------------------------------------------
    std::cout << "[DAEMON] Servicio activo. Ctrl+C para detener." << std::endl;

    cv::Mat frame;
    int emptyFrameCount = 0;
    while (keepRunning.load()) {
      // --- Sondeo IPC (ANTES de lectura de hardware) ---
      if (ipcServer->pollCommand(asyncCmdByte)) {
        processIpcCommand(asyncCmdByte, effectEnabled);
      }

      cap.read(frame);
      if (frame.empty()) {
        emptyFrameCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (emptyFrameCount >= 90) {
          throw std::runtime_error(
              "Falla cr\u00edtica: Hardware de video inoperante por 3 "
              "segundos. Abortando proceso.");
        }
        continue;
      }
      emptyFrameCount = 0;

      auto start = std::chrono::high_resolution_clock::now();

      vision.process(frame, effectEnabled);

      videoSink->writeFrame(frame);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_latency += elapsed.count();
      frames_processed++;
    }

  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  // -----------------------------------------------------------------
  // Shutdown: higiene de descriptores
  // -----------------------------------------------------------------
  std::cout << "\n[DAEMON] Iniciando shutdown limpio..." << std::endl;

  // IpcServer release and closing handles is managed by std::unique_ptr DAII
  ipcServer.reset();

  if (frames_processed > 0) {
    std::cout << "[RESULTADO] Frames totales: " << frames_processed
              << std::endl;
    std::cout << "[RESULTADO] Latencia promedio: "
              << (total_latency / frames_processed) << " ms" << std::endl;
  }

  size_t memFinal = getProcessMemory();
  std::cout << "[MEMORIA] Fin del pipeline: " << (memFinal / (1024.0 * 1024.0))
            << " MB" << std::endl;

  cap.release();
  std::cout << "[DAEMON] Shutdown limpio completado." << std::endl;
  return 0;
}

#else

// =====================================================================
// ARQUITECTURA LINUX (v4l2loopback + Unix Domain Socket IPC)
// =====================================================================

int main() {
  // --- Registro de señales ---
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::cout << "=== DirectLook Daemon [Linux] ===" << std::endl;
  std::cout << "[DAEMON] Modo servicio continuo activo." << std::endl;

  // Inicializar rutas dinámicas con resolución heurística
  MODEL_PATH = resolveModelPath("pfld.onnx");
  FACE_MODEL_PATH = resolveModelPath("version-slim-320_simplified.onnx");

  // Variables para limpieza garantizada
  std::unique_ptr<IpcServer> ipcServer = std::make_unique<UnixSocketServer>();
  std::unique_ptr<VideoSink> videoSink =
      std::make_unique<LinuxV4l2Sink>("/dev/video2");

  cv::VideoCapture cap;
  size_t frames_processed = 0;
  double total_latency = 0.0;

  try {
    size_t memInicio = getProcessMemory();
    std::cout << "[MEMORIA] Inicio del proceso: "
              << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

    // -----------------------------------------------------------------
    // Instanciación del Motor ONNX Runtime (CPU)
    // -----------------------------------------------------------------
    VisionPipeline vision(FACE_MODEL_PATH, MODEL_PATH);

    // -----------------------------------------------------------------
    // Captura de video + inyección a sumidero configurado
    // -----------------------------------------------------------------
    cap.open(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
      throw std::runtime_error(
          "Falla estructural: Imposible adquirir /dev/video0.");
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    cap.set(cv::CAP_PROP_FPS, 15);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    bool effectEnabled = true;

    // -----------------------------------------------------------------
    // Bucle principal perpetuo
    // -----------------------------------------------------------------
    std::cout << "[DAEMON] Servicio activo. kill -SIGINT <pid> para detener."
              << std::endl;

    cv::Mat frame;
    int emptyFrameCount = 0;
    while (keepRunning.load()) {
      // --- Sondeo IPC (ANTES de lectura de hardware) ---
      uint8_t cmdByte;
      if (ipcServer->pollCommand(cmdByte)) {
        processIpcCommand(cmdByte, effectEnabled);
      }

      cap.read(frame);
      if (frame.empty()) {
        emptyFrameCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (emptyFrameCount >= 90) {
          throw std::runtime_error(
              "Falla cr\u00edtica: Hardware de video inoperante por 3 "
              "segundos. Abortando proceso.");
        }
        continue;
      }
      emptyFrameCount = 0;

      auto start = std::chrono::high_resolution_clock::now();

      vision.process(frame, effectEnabled);

      videoSink->writeFrame(frame);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_latency += elapsed.count();
      frames_processed++;
    }
  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  // -----------------------------------------------------------------
  // Shutdown: higiene de descriptores y desvinculado de sockets
  // -----------------------------------------------------------------
  std::cout << "\n[DAEMON] Iniciando shutdown limpio..." << std::endl;

  // IpcServer release and closing descriptors is managed by std::unique_ptr
  // RAII
  ipcServer.reset();

  if (frames_processed > 0) {
    std::cout << "[RESULTADO] Frames totales: " << frames_processed
              << std::endl;
    std::cout << "[RESULTADO] Latencia promedio: "
              << (total_latency / frames_processed) << " ms" << std::endl;
  }

  size_t memFinal = getProcessMemory();
  std::cout << "[MEMORIA] Fin del pipeline: " << (memFinal / (1024.0 * 1024.0))
            << " MB" << std::endl;

  if (cap.isOpened())
    cap.release();
  std::cout << "[DAEMON] Shutdown limpio completado." << std::endl;
  return 0;
}
#endif