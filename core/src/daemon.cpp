// =============================================================================
// DirectLook — Daemon Core (Multiplataforma: Windows + Linux)
// Sprint 2: Servicio Continuo · Canal IPC · Protocolo de Telemetría
// =============================================================================

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

#include "InpaintingEngine.hpp"
#include "cpu_monitor.h"
#include "ipc_server.h"
#include "protocol.h"
#include "vision_pipeline.h"

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

std::atomic<bool> keepRunning{true};

void signalHandler(int signum) {
  keepRunning.store(false);
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
  std::vector<std::string> searchPaths = {
      exeDir + "/modelos/" + modelName, exeDir + "/../modelos/" + modelName,
      exeDir + "/../../modelos/" + modelName};

  for (const auto &path : searchPaths) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f) {
      fclose(f);
      return path;
    }
  }
  return exeDir + "/modelos/" + modelName;
}

static constexpr size_t MEMORY_LIMIT_BYTES = 125 * 1024 * 1024;
std::string MODEL_PATH;
std::string FACE_MODEL_PATH;

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

#ifdef _WIN32
// =====================================================================
// ARQUITECTURA WINDOWS (DirectShow / MSMF + Named Pipe IPC)
// =====================================================================
int main(int argc, char **argv) {
  int cameraIndex = 0;
  int targetFps = 30;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--camera" && i + 1 < argc) {
      cameraIndex = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--limit-fps" && i + 1 < argc) {
      targetFps = std::stoi(argv[++i]);
    }
  }
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::cout << "=== DirectLook Daemon [Windows] ===" << std::endl;
  MODEL_PATH = resolveModelPath("pfld.onnx");
  FACE_MODEL_PATH = resolveModelPath("version-slim-320_simplified.onnx");
  std::string INPAINT_MODEL_PATH = resolveModelPath("inpainting_fp32.onnx");

  InpaintingEngine inpainter(INPAINT_MODEL_PATH);

  std::unique_ptr<IpcServer> ipcServer =
      std::make_unique<WindowsNamedPipeServer>();
  std::unique_ptr<VideoSink> videoSink =
      std::make_unique<WindowsVirtualCamSink>();

  cv::VideoCapture cap;
  size_t frames_processed = 0;
  double total_latency = 0.0;

  try {
    cap.open(cameraIndex, cv::CAP_DSHOW);
    if (!cap.isOpened())
      throw std::runtime_error("Falla estructural: Cámara inaccesible.");

    cap.set(cv::CAP_PROP_FPS, targetFps);
    double actualFps = cap.get(cv::CAP_PROP_FPS);
    if (actualFps <= 0)
      actualFps = targetFps;

    VisionPipeline vision(FACE_MODEL_PATH, MODEL_PATH, actualFps);

    bool effectEnabled = true;
    uint8_t asyncCmdByte = 0;
    CpuMonitor monitor;
    cv::Mat frame;
    int emptyFrameCount = 0;

    std::thread ipcWatchdog([&monitor, &ipcServer]() {
      bool alarmSent = false;
      while (keepRunning.load()) {
        int level = monitor.getDegradationLevel();
        if (level == 3 && !alarmSent) {
          ipcServer->sendTelemetry(DIRECTLOOK_CMD_THERMAL_ALARM);
          alarmSent = true;
        } else if (level < 3)
          alarmSent = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });

    while (keepRunning.load()) {
      if (ipcServer->pollCommand(asyncCmdByte))
        processIpcCommand(asyncCmdByte, effectEnabled);

      cap.read(frame);
      if (frame.empty()) {
        emptyFrameCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (emptyFrameCount >= 90)
          throw std::runtime_error("Falla crítica de hardware.");
        continue;
      }
      emptyFrameCount = 0;

      cv::resize(frame, frame, cv::Size(640, 360));
      auto start = std::chrono::high_resolution_clock::now();
      int level = monitor.getDegradationLevel();

      vision.process(frame, effectEnabled, level);

      if (effectEnabled && level < 3 && vision.hasValidEyes()) {
        cv::Rect leftRoi = vision.getLeftEyeRoi();
        cv::Rect rightRoi = vision.getRightEyeRoi();

        if (leftRoi.area() > 0 && rightRoi.area() > 0) {
          cv::Mat leftCrop = frame(leftRoi);
          cv::Mat rightCrop = frame(rightRoi);

          cv::Mat newLeftEye = inpainter.processEye(leftCrop);
          cv::Mat newRightEye = inpainter.processEye(rightCrop);

          if (!newLeftEye.empty() && newLeftEye.size() == leftRoi.size()) {
            newLeftEye.copyTo(frame(leftRoi));
          }
          if (!newRightEye.empty() && newRightEye.size() == rightRoi.size()) {
            newRightEye.copyTo(frame(rightRoi));
          }
        }
      }

      videoSink->writeFrame(frame);
      auto end = std::chrono::high_resolution_clock::now();
      total_latency +=
          std::chrono::duration<double, std::milli>(end - start).count();
      frames_processed++;
    }
    keepRunning.store(false);
    if (ipcWatchdog.joinable())
      ipcWatchdog.join();
  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  ipcServer.reset();
  cap.release();
  return 0;
}

#else
// =====================================================================
// ARQUITECTURA LINUX (v4l2loopback + Unix Domain Socket IPC)
// =====================================================================
int main(int argc, char **argv) {
  int targetFps = 30;
  std::string videoSource = "";
  std::string videoSinkPath = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--video-source=") == 0)
      videoSource = arg.substr(15);
    else if (arg == "--video-source" && i + 1 < argc)
      videoSource = argv[++i];
    else if (arg.find("--video-sink=") == 0)
      videoSinkPath = arg.substr(13);
    else if (arg == "--video-sink" && i + 1 < argc)
      videoSinkPath = argv[++i];
    else if (arg.find("--limit-fps=") == 0)
      targetFps = std::stoi(arg.substr(12));
    else if (arg == "--limit-fps" && i + 1 < argc)
      targetFps = std::stoi(argv[++i]);
  }

  if (videoSource.empty() || videoSinkPath.empty()) {
    throw std::runtime_error(
        "Argumentos CLI requeridos ausentes: --video-source y --video-sink.");
  }
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::cout << "=== DirectLook Daemon [Linux] ===" << std::endl;
  MODEL_PATH = resolveModelPath("pfld.onnx");
  FACE_MODEL_PATH = resolveModelPath("version-slim-320_simplified.onnx");
  std::string INPAINT_MODEL_PATH = resolveModelPath("inpainting_fp32.onnx");

  InpaintingEngine inpainter(INPAINT_MODEL_PATH);

  std::unique_ptr<IpcServer> ipcServer = std::make_unique<UnixSocketServer>();
  std::unique_ptr<VideoSink> videoSink =
      std::make_unique<LinuxV4l2Sink>(videoSinkPath);

  cv::VideoCapture cap;
  size_t frames_processed = 0;
  double total_latency = 0.0;

  try {
    cap.open(videoSource, cv::CAP_V4L2);
    if (!cap.isOpened())
      throw std::runtime_error(
          "Falla estructural: Descriptor fuente inaccesible.");

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    cap.set(cv::CAP_PROP_FPS, targetFps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    double actualFps = cap.get(cv::CAP_PROP_FPS);
    if (actualFps <= 0)
      actualFps = targetFps;

    VisionPipeline vision(FACE_MODEL_PATH, MODEL_PATH, actualFps);

    bool effectEnabled = true;
    CpuMonitor monitor;
    cv::Mat frame;
    int emptyFrameCount = 0;

    std::thread ipcWatchdog([&monitor, &ipcServer]() {
      bool alarmSent = false;
      while (keepRunning.load()) {
        int level = monitor.getDegradationLevel();
        if (level == 3 && !alarmSent) {
          ipcServer->sendTelemetry(DIRECTLOOK_CMD_THERMAL_ALARM);
          alarmSent = true;
        } else if (level < 3)
          alarmSent = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });

    while (keepRunning.load()) {
      uint8_t cmdByte;
      if (ipcServer->pollCommand(cmdByte))
        processIpcCommand(cmdByte, effectEnabled);

      cap.read(frame);
      if (frame.empty()) {
        emptyFrameCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (emptyFrameCount >= 90)
          throw std::runtime_error("Falla crítica de video.");
        continue;
      }
      emptyFrameCount = 0;

      cv::resize(frame, frame, cv::Size(640, 360));
      auto start = std::chrono::high_resolution_clock::now();
      int level = monitor.getDegradationLevel();

      vision.process(frame, effectEnabled, level);

      if (effectEnabled && level < 3 && vision.hasValidEyes()) {
        cv::Rect leftRoi = vision.getLeftEyeRoi();
        cv::Rect rightRoi = vision.getRightEyeRoi();

        if (leftRoi.area() > 0 && rightRoi.area() > 0) {
          cv::Mat leftCrop = frame(leftRoi);
          cv::Mat rightCrop = frame(rightRoi);

          cv::Mat newLeftEye = inpainter.processEye(leftCrop);
          cv::Mat newRightEye = inpainter.processEye(rightCrop);

          if (!newLeftEye.empty() && newLeftEye.size() == leftRoi.size()) {
            newLeftEye.copyTo(frame(leftRoi));
          }
          if (!newRightEye.empty() && newRightEye.size() == rightRoi.size()) {
            newRightEye.copyTo(frame(rightRoi));
          }
        }
      }

      videoSink->writeFrame(frame);
      auto end = std::chrono::high_resolution_clock::now();
      total_latency +=
          std::chrono::duration<double, std::milli>(end - start).count();
      frames_processed++;
    }
    keepRunning.store(false);
    if (ipcWatchdog.joinable())
      ipcWatchdog.join();
  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  ipcServer.reset();
  if (cap.isOpened())
    cap.release();
  return 0;
}
#endif