// =============================================================================
// DirectLook — Daemon Core (Multiplataforma: Windows + Linux)
// Sprint 2: Servicio Continuo · Canal IPC · Protocolo de Telemetría
// =============================================================================

// --- Includes comunes (agnósticos de plataforma) ---
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "protocol.h"

// --- Includes condicionales de plataforma ---

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

#include <psapi.h>
#else

#include <fcntl.h>
#include <fstream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
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

// Dimensiones exigidas por el modelo PFLD (112x112)
static constexpr int MODEL_SIZE = 112;
static constexpr int TENSOR_ELEMENTS = 1 * 3 * MODEL_SIZE * MODEL_SIZE;

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
// Transmutación de Tensores: BGR/HWC → RGB/NCHW float [0.0, 1.0]
// =============================================================================
void preprocessFrame(const cv::Mat &frame, cv::Mat &resized, float *buffer) {
  cv::resize(frame, resized, cv::Size(MODEL_SIZE, MODEL_SIZE));
  cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(),
                                        cv::Scalar(), true);
  std::memcpy(buffer, blob.ptr<float>(), blob.total() * sizeof(float));
}

// =============================================================================
// Generador de Prior Boxes para UltraFace SSD (slim-320)
// Produce 4420 anchors basados en feature maps [30x40, 15x20, 8x10, 4x5]
// =============================================================================
std::vector<std::array<float, 4>> generateUltraFacePriors(int imgW, int imgH) {
  std::vector<std::array<float, 4>> priors;
  const int strides[] = {8, 16, 32, 64};
  const std::vector<std::vector<float>> min_boxes = {{10.0f, 16.0f, 24.0f},
                                                     {32.0f, 48.0f},
                                                     {64.0f, 96.0f},
                                                     {128.0f, 192.0f, 256.0f}};

  for (int i = 0; i < 4; ++i) {
    int fmH =
        static_cast<int>(std::ceil(static_cast<float>(imgH) / strides[i]));
    int fmW =
        static_cast<int>(std::ceil(static_cast<float>(imgW) / strides[i]));
    for (int y = 0; y < fmH; ++y) {
      for (int x = 0; x < fmW; ++x) {
        for (float mb : min_boxes[i]) {
          float cx = (x + 0.5f) * strides[i] / static_cast<float>(imgW);
          float cy = (y + 0.5f) * strides[i] / static_cast<float>(imgH);
          float w = mb / static_cast<float>(imgW);
          float h = mb / static_cast<float>(imgH);
          priors.push_back({cx, cy, w, h});
        }
      }
    }
  }
  return priors;
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
  HANDLE hPipe = INVALID_HANDLE_VALUE;
  OVERLAPPED olConnect;
  OVERLAPPED olRead;
  std::memset(&olConnect, 0, sizeof(olConnect));
  std::memset(&olRead, 0, sizeof(olRead));
  bool pipeConnected = false;
  bool readPending = false;
  bool connectPending = false;
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
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");
    Ort::SessionOptions sessionOpts;

    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOpts.DisableCpuMemArena();
    sessionOpts.DisableMemPattern();

    std::unique_ptr<Ort::Session> session;
    bool modelLoaded = false;

    try {
      std::wstring wpath(MODEL_PATH.begin(), MODEL_PATH.end());
      session = std::make_unique<Ort::Session>(env, wpath.c_str(), sessionOpts);
      modelLoaded = true;
      std::cout << "[ONNX] Modelo cargado: " << MODEL_PATH << std::endl;
      // --- Introspección del modelo: dimensiones exactas ---
      Ort::AllocatorWithDefaultOptions tmpAlloc;
      size_t numIn = session->GetInputCount();
      size_t numOut = session->GetOutputCount();
      for (size_t i = 0; i < numIn; ++i) {
        auto name = session->GetInputNameAllocated(i, tmpAlloc);
        auto info = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        std::cout << "[ONNX]   Input[" << i << "]: '" << name.get()
                  << "' shape=[";
        for (size_t s = 0; s < shape.size(); ++s)
          std::cout << shape[s] << (s < shape.size() - 1 ? "," : "");
        std::cout << "]" << std::endl;
      }
      for (size_t i = 0; i < numOut; ++i) {
        auto name = session->GetOutputNameAllocated(i, tmpAlloc);
        auto info = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        std::cout << "[ONNX]   Output[" << i << "]: '" << name.get()
                  << "' shape=[";
        for (size_t s = 0; s < shape.size(); ++s)
          std::cout << shape[s] << (s < shape.size() - 1 ? "," : "");
        std::cout << "]" << std::endl;
      }
      // --- Auditoría de memoria post-carga ---
      size_t memPostCarga = getProcessMemory();
      double memMB = memPostCarga / (1024.0 * 1024.0);
      std::cout << "[MEMORIA] Post-carga ONNX: " << memMB << " MB" << std::endl;
      if (memPostCarga > MEMORY_LIMIT_BYTES) {
        throw std::runtime_error("Límite arquitectónico de 80MB excedido (" +
                                 std::to_string(memMB) + " MB)");
      }

    } catch (const Ort::Exception &e) {
      std::cerr << "[ONNX] No se pudo cargar modelo: " << e.what() << std::endl;
      std::cout << "[ONNX] Continuando en modo benchmark (sin inferencia)."
                << std::endl;
    }

    // -----------------------------------------------------------------
    // Captura de video
    // -----------------------------------------------------------------
    cap.open(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
      std::cerr << "Falla estructural: Imposible adquirir cámara en Windows."
                << std::endl;
      return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    cap.set(cv::CAP_PROP_FPS, 15);

    // -----------------------------------------------------------------
    // Detector facial UltraFace ONNX
    // -----------------------------------------------------------------
    static constexpr int UF_W = 320, UF_H = 240;
    static constexpr int UF_ELEMENTS = 1 * 3 * UF_H * UF_W;
    std::unique_ptr<Ort::Session> faceSession;
    bool faceDetectorOk = false;

    try {
      std::wstring fwpath(FACE_MODEL_PATH.begin(), FACE_MODEL_PATH.end());
      faceSession =
          std::make_unique<Ort::Session>(env, fwpath.c_str(), sessionOpts);
      faceDetectorOk = true;
      std::cout << "[FACE] UltraFace cargado: " << FACE_MODEL_PATH << std::endl;
    } catch (const Ort::Exception &e) {
      std::cerr << "[FACE] No se pudo cargar UltraFace: " << e.what()
                << std::endl;
    }

    // Buffers UltraFace pre-alocados
    std::vector<float> ufTensorData(UF_ELEMENTS);
    std::array<int64_t, 4> ufShape = {1, 3, UF_H, UF_W};
    Ort::Value ufInputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        ufTensorData.data(), ufTensorData.size(), ufShape.data(),
        ufShape.size());

    // Nombres cacheados para UltraFace
    const char *ufInputName = nullptr;
    std::vector<const char *> ufOutputNames;
    Ort::AllocatedStringPtr ufInName(nullptr,
                                     Ort::detail::AllocatedFree{nullptr});
    std::vector<Ort::AllocatedStringPtr> ufOutNames;
    // Priors para decodificación SSD (generados una sola vez)
    auto ufPriors = generateUltraFacePriors(UF_W, UF_H);
    std::cout << "[FACE] Priors generados: " << ufPriors.size() << std::endl;

    if (faceDetectorOk && faceSession) {
      Ort::AllocatorWithDefaultOptions ufAlloc;
      ufInName = faceSession->GetInputNameAllocated(0, ufAlloc);
      ufInputName = ufInName.get();
      for (size_t i = 0; i < faceSession->GetOutputCount(); ++i) {
        ufOutNames.push_back(faceSession->GetOutputNameAllocated(i, ufAlloc));
        ufOutputNames.push_back(ufOutNames.back().get());
      }
    }

    // -----------------------------------------------------------------
    // Buffers PFLD pre-alocados
    // -----------------------------------------------------------------
    cv::Mat frame;
    cv::Mat resized;
    cv::Mat ufResized;
    // double total_latency = 0.0; // Duplicate declaration, removed
    // int frames_processed = 0; // Duplicate declaration, removed

    std::vector<float> tensorData(TENSOR_ELEMENTS);
    std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
    auto memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char *cachedInputName = nullptr;
    const char *cachedOutputName = nullptr;
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::AllocatedStringPtr inNameHolder(nullptr,
                                         Ort::detail::AllocatedFree{nullptr});
    Ort::AllocatedStringPtr outNameHolder(nullptr,
                                          Ort::detail::AllocatedFree{nullptr});
    Ort::RunOptions runOpts;

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
        inputShape.size());

    if (modelLoaded && session) {
      inNameHolder = session->GetInputNameAllocated(0, alloc);
      outNameHolder = session->GetOutputNameAllocated(0, alloc);
      cachedInputName = inNameHolder.get();
      cachedOutputName = outNameHolder.get();
    }

    // -----------------------------------------------------------------
    // Canal IPC: Named Pipe (Overlapped I/O asíncrono nativo)
    // -----------------------------------------------------------------
    hPipe = CreateNamedPipeA(DIRECTLOOK_PIPE_NAME,
                             PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                             PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                             1, // instancias máximas
                             1, // buffer salida (1 byte)
                             1, // buffer entrada (1 byte)
                             0, // timeout
                             NULL);

    pipeConnected = false;
    readPending = false;
    bool effectEnabled = true;
    uint8_t asyncCmdByte = 0;

    olConnect.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    olRead.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    connectPending = false;

    if (hPipe == INVALID_HANDLE_VALUE) {
      throw std::runtime_error(
          "Falla cr\u00edtica: Imposible instanciar Named Pipe IPC.");
    } else {
      std::cout << "[IPC] Named Pipe creado: " << DIRECTLOOK_PIPE_NAME
                << std::endl;
      // Iniciar escucha asíncrona de conexión
      ConnectNamedPipe(hPipe, &olConnect);
      DWORD err = GetLastError();
      if (err == ERROR_IO_PENDING) {
        connectPending = true;
      } else if (err == ERROR_PIPE_CONNECTED) {
        pipeConnected = true;
        std::cout << "[IPC] Cliente conectado." << std::endl;
      }
    }

    // -----------------------------------------------------------------
    // Bucle principal perpetuo
    // -----------------------------------------------------------------
    std::cout << "[DAEMON] Servicio activo. Ctrl+C para detener." << std::endl;

    while (keepRunning.load()) {
      // --- Sondeo IPC (ANTES de lectura de hardware) ---
      if (hPipe != INVALID_HANDLE_VALUE) {
        // Verificar conexión pendiente
        if (!pipeConnected && connectPending) {
          DWORD dummy;
          if (GetOverlappedResult(hPipe, &olConnect, &dummy, FALSE)) {
            pipeConnected = true;
            connectPending = false;
            std::cout << "[IPC] Cliente conectado." << std::endl;
          }
        }

        if (pipeConnected) {
          if (!readPending) {
            // Iniciar lectura asíncrona de 1 byte
            DWORD bytesRead = 0;
            if (ReadFile(hPipe, &asyncCmdByte, 1, &bytesRead, &olRead)) {
              // Completado sincrónicamente
              processIpcCommand(asyncCmdByte, effectEnabled);
            } else if (GetLastError() == ERROR_IO_PENDING) {
              readPending = true;
            } else {
              // Cliente desconectado
              DisconnectNamedPipe(hPipe);
              pipeConnected = false;
              readPending = false;
              std::cout << "[IPC] Cliente desconectado. Esperando reconexión..."
                        << std::endl;
              ResetEvent(olConnect.hEvent);
              ConnectNamedPipe(hPipe, &olConnect);
              connectPending = (GetLastError() == ERROR_IO_PENDING);
            }
          } else {
            // Sondear lectura pendiente (bWait=FALSE, zero-block)
            DWORD bytesRead = 0;
            if (GetOverlappedResult(hPipe, &olRead, &bytesRead, FALSE)) {
              readPending = false;
              if (bytesRead == 1) {
                processIpcCommand(asyncCmdByte, effectEnabled);
              }
            } else if (GetLastError() != ERROR_IO_INCOMPLETE) {
              // Error fatal de lectura → desconexión
              DisconnectNamedPipe(hPipe);
              pipeConnected = false;
              readPending = false;
              std::cout << "[IPC] Cliente desconectado. Esperando reconexión..."
                        << std::endl;
              ResetEvent(olConnect.hEvent);
              ConnectNamedPipe(hPipe, &olConnect);
              connectPending = (GetLastError() == ERROR_IO_PENDING);
            }
          }
        }
      }

      cap.read(frame);
      if (frame.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }

      auto start = std::chrono::high_resolution_clock::now();

      // --- Pipeline de visión (solo si efecto habilitado) ---
      cv::Rect roi;
      bool faceFound = false;

      if (effectEnabled && faceDetectorOk && faceSession) {
        // [PASO 1] Preprocesar para UltraFace: resize 320x240, (px-127)/128,
        // BGR
        cv::resize(frame, ufResized, cv::Size(UF_W, UF_H));
        const int ufPlane = UF_H * UF_W;
        float *ch0 = ufTensorData.data();
        float *ch1 = ch0 + ufPlane;
        float *ch2 = ch1 + ufPlane;
        for (int y = 0; y < UF_H; ++y) {
          const uint8_t *row = ufResized.ptr<uint8_t>(y);
          for (int x = 0; x < UF_W; ++x) {
            int idx = y * UF_W + x;
            int px3 = x * 3;
            ch0[idx] = (row[px3 + 0] - 127.0f) / 128.0f; // B
            ch1[idx] = (row[px3 + 1] - 127.0f) / 128.0f; // G
            ch2[idx] = (row[px3 + 2] - 127.0f) / 128.0f; // R
          }
        }

        try {
          auto ufResults =
              faceSession->Run(runOpts, &ufInputName, &ufInputTensor, 1,
                               ufOutputNames.data(), ufOutputNames.size());

          const float *scores = ufResults[0].GetTensorData<float>();
          const float *boxes = ufResults[1].GetTensorData<float>();
          int numAnchors = static_cast<int>(ufPriors.size());

          float bestConf = 0.7f;
          int bestIdx = -1;
          for (int a = 0; a < numAnchors; ++a) {
            float conf = scores[a * 2 + 1]; // [bg, face]
            if (conf > bestConf) {
              bestConf = conf;
              bestIdx = a;
            }
          }

          if (bestIdx >= 0) {
            const float CENTER_VAR = 0.1f;
            const float SIZE_VAR = 0.2f;
            float pcx = ufPriors[bestIdx][0];
            float pcy = ufPriors[bestIdx][1];
            float pw = ufPriors[bestIdx][2];
            float ph = ufPriors[bestIdx][3];

            float cx = pcx + boxes[bestIdx * 4 + 0] * CENTER_VAR * pw;
            float cy = pcy + boxes[bestIdx * 4 + 1] * CENTER_VAR * ph;
            float bw = pw * std::exp(boxes[bestIdx * 4 + 2] * SIZE_VAR);
            float bh = ph * std::exp(boxes[bestIdx * 4 + 3] * SIZE_VAR);

            float bx1 = cx - bw / 2.0f;
            float by1 = cy - bh / 2.0f;
            float bx2 = cx + bw / 2.0f;
            float by2 = cy + bh / 2.0f;

            int x1 = static_cast<int>(bx1 * frame.cols);
            int y1 = static_cast<int>(by1 * frame.rows);
            int x2 = static_cast<int>(bx2 * frame.cols);
            int y2 = static_cast<int>(by2 * frame.rows);

            int w = x2 - x1, h = y2 - y1;
            if (w > 10 && h > 10) {
              int padW = w / 10, padH = h / 10;
              x1 = std::max(0, x1 - padW);
              y1 = std::max(0, y1 - padH);
              x2 = std::min(frame.cols, x2 + padW);
              y2 = std::min(frame.rows, y2 + padH);
              roi = cv::Rect(x1, y1, x2 - x1, y2 - y1);
              faceFound = true;
            }
          }
        } catch (const Ort::Exception &e) {
          std::cerr << "[FACE] Error UltraFace: " << e.what() << std::endl;
        }
      }

      if (faceFound) {
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // [PASO 2] Recortar cara y preprocesar para PFLD
        cv::Mat faceCrop = frame(roi);
        preprocessFrame(faceCrop, resized, tensorData.data());

        // [PASO 3] Inferencia ONNX (PFLD: 98 landmarks)
        if (modelLoaded && session) {
          try {
            auto results = session->Run(runOpts, &cachedInputName, &inputTensor,
                                        1, &cachedOutputName, 1);

            const float *landmarks = results[0].GetTensorData<float>();

            for (int p = 0; p < 98; ++p) {
              float pfld_x = landmarks[p * 2];
              float pfld_y = landmarks[p * 2 + 1];

              int final_x = roi.x + static_cast<int>(pfld_x * roi.width);
              int final_y = roi.y + static_cast<int>(pfld_y * roi.height);

              bool isEyeOrPupil = (p >= 60 && p <= 75) || (p >= 96 && p <= 97);
              cv::Scalar color =
                  isEyeOrPupil ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
              int radius = isEyeOrPupil ? 3 : 2;

              cv::circle(frame, cv::Point(final_x, final_y), radius, color, -1,
                         cv::LINE_AA);
            }

          } catch (const Ort::Exception &e) {
            std::cerr << "[ONNX] Error inferencia: " << e.what() << std::endl;
          }
        }
      } else if (effectEnabled) {
        cv::putText(frame, "ESTADO: BUSCANDO ROSTRO...", cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2,
                    cv::LINE_AA);
      }

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_latency += elapsed.count();
      frames_processed++;

      // Reporte periódico cada 300 frames
      if (frames_processed % 300 == 0) {
        std::cout << "[PIPELINE] Frames: " << frames_processed
                  << " | Latencia promedio: "
                  << (total_latency / frames_processed) << " ms"
                  << " | RAM: " << (getProcessMemory() / (1024.0 * 1024.0))
                  << " MB"
                  << " | Efecto: " << (effectEnabled ? "ON" : "OFF")
                  << std::endl;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  // -----------------------------------------------------------------
  // Shutdown: higiene de descriptores
  // -----------------------------------------------------------------
  std::cout << "\n[DAEMON] Iniciando shutdown limpio..." << std::endl;

  if (hPipe != INVALID_HANDLE_VALUE) {
    if (readPending)
      CancelIo(hPipe);
    if (pipeConnected)
      DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);
    if (olConnect.hEvent)
      CloseHandle(olConnect.hEvent);
    if (olRead.hEvent)
      CloseHandle(olRead.hEvent);
    std::cout << "[IPC] Named Pipe cerrado." << std::endl;
  }

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
  cv::VideoCapture cap;
  int fd = -1;
  int sockFd = -1;
  int clientFd = -1;
  size_t frames_processed = 0;
  double total_latency = 0.0;

  try {
    size_t memInicio = getProcessMemory();
    std::cout << "[MEMORIA] Inicio del proceso: "
              << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

    // -----------------------------------------------------------------
    // Instanciación del Motor ONNX Runtime (CPU)
    // -----------------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");

    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOpts.DisableCpuMemArena();
    sessionOpts.DisableMemPattern();

    std::unique_ptr<Ort::Session> session;
    bool modelLoaded = false;

    // ... (ONNX load try/catch is inside the main try/catch) ...
    try {
      session =
          std::make_unique<Ort::Session>(env, MODEL_PATH.c_str(), sessionOpts);
      modelLoaded = true;
      std::cout << "[ONNX] Modelo cargado: " << MODEL_PATH << std::endl;

      size_t memPostCarga = getProcessMemory();
      double memMB = memPostCarga / (1024.0 * 1024.0);
      std::cout << "[MEMORIA] Post-carga ONNX: " << memMB << " MB" << std::endl;

      if (memPostCarga > MEMORY_LIMIT_BYTES) {
        throw std::runtime_error("Límite arquitectónico de 80MB excedido (" +
                                 std::to_string(memMB) + " MB)");
      }

    } catch (const Ort::Exception &e) {
      std::cerr << "[ONNX] No se pudo cargar modelo: " << e.what() << std::endl;
      std::cout << "[ONNX] Continuando en modo benchmark (sin inferencia)."
                << std::endl;
    }

    // -----------------------------------------------------------------
    // Captura de video + cámara virtual (/dev/video2)
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

    fd = open("/dev/video2", O_RDWR);
    if (fd < 0) {
      throw std::runtime_error(
          "Falla de kernel: Descriptor /dev/video2 inaccesible.");
    }

    struct v4l2_format fmt;
    std::memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 360;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
    fmt.fmt.pix.sizeimage = 640 * 360 * 3;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      throw std::runtime_error(
          "Falla de I/O: Imposible negociar formato en descriptor virtual.");
    }

    // -----------------------------------------------------------------
    // Detector facial UltraFace ONNX
    // -----------------------------------------------------------------
    std::unique_ptr<Ort::Session> faceSession;
    bool faceDetectorOk = false;

    try {
      faceSession = std::make_unique<Ort::Session>(env, FACE_MODEL_PATH.c_str(),
                                                   sessionOpts);
      faceDetectorOk = true;
      std::cout << "[FACE] UltraFace cargado: " << FACE_MODEL_PATH << std::endl;
    } catch (const Ort::Exception &e) {
      std::cerr << "[FACE] No se pudo cargar UltraFace: " << e.what()
                << std::endl;
    }

    const int UF_W = 320;
    const int UF_H = 240;
    std::vector<float> ufTensorData(1 * 3 * UF_H * UF_W);
    std::array<int64_t, 4> ufShape = {1, 3, UF_H, UF_W};
    auto ufMemInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value ufInputTensor = Ort::Value::CreateTensor<float>(
        ufMemInfo, ufTensorData.data(), ufTensorData.size(), ufShape.data(),
        ufShape.size());

    const char *ufInputName = nullptr;
    std::vector<const char *> ufOutputNames;
    Ort::AllocatedStringPtr ufInName(nullptr,
                                     Ort::detail::AllocatedFree{nullptr});
    std::vector<Ort::AllocatedStringPtr> ufOutNames;

    auto ufPriors = generateUltraFacePriors(UF_W, UF_H);

    if (faceDetectorOk && faceSession) {
      Ort::AllocatorWithDefaultOptions ufAlloc;
      ufInName = faceSession->GetInputNameAllocated(0, ufAlloc);
      ufInputName = ufInName.get();

      for (size_t i = 0; i < faceSession->GetOutputCount(); ++i) {
        ufOutNames.push_back(faceSession->GetOutputNameAllocated(i, ufAlloc));
        ufOutputNames.push_back(ufOutNames.back().get());
      }
    }

    // -----------------------------------------------------------------
    // Buffers PFLD pre-alocados
    // -----------------------------------------------------------------
    cv::Mat frame;
    cv::Mat resized;
    cv::Mat ufResized;
    double total_latency = 0.0;
    int frames_processed = 0;

    std::vector<float> tensorData(TENSOR_ELEMENTS);
    std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
    auto memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char *cachedInputName = nullptr;
    const char *cachedOutputName = nullptr;
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::AllocatedStringPtr inNameHolder(nullptr,
                                         Ort::detail::AllocatedFree{nullptr});
    Ort::AllocatedStringPtr outNameHolder(nullptr,
                                          Ort::detail::AllocatedFree{nullptr});
    Ort::RunOptions runOpts;

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
        inputShape.size());

    if (modelLoaded && session) {
      inNameHolder = session->GetInputNameAllocated(0, alloc);
      outNameHolder = session->GetOutputNameAllocated(0, alloc);
      cachedInputName = inNameHolder.get();
      cachedOutputName = outNameHolder.get();
    }

    // -----------------------------------------------------------------
    // Canal IPC: Socket de Dominio UNIX (AF_UNIX, no bloqueante)
    // -----------------------------------------------------------------
    sockFd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    clientFd = -1;
    bool effectEnabled = true;

    if (sockFd < 0) {
      throw std::runtime_error(
          "Falla cr\u00edtica: Imposible instanciar Socket UNIX IPC.");
    }

    // Limpiar socket obsoleto antes de bind
    unlink(DIRECTLOOK_SOCK_PATH);

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, DIRECTLOOK_SOCK_PATH,
                 sizeof(addr.sun_path) - 1);

    if (bind(sockFd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
        0) {
      throw std::runtime_error(
          "Falla cr\u00edtica: Imposible instanciar Socket UNIX IPC.");
    }
    if (listen(sockFd, 1) < 0) {
      throw std::runtime_error(
          "Falla cr\u00edtica: Imposible instanciar Socket UNIX IPC.");
    }
    std::cout << "[IPC] Socket UNIX creado: " << DIRECTLOOK_SOCK_PATH
              << std::endl;

    // -----------------------------------------------------------------
    // Bucle principal perpetuo
    // -----------------------------------------------------------------
    std::cout << "[DAEMON] Servicio activo. kill -SIGINT <pid> para detener."
              << std::endl;

    while (keepRunning.load()) {
      // --- Sondeo IPC (ANTES de lectura de hardware) ---
      if (sockFd >= 0) {
        // Aceptar nuevo cliente (no bloqueante)
        if (clientFd < 0) {
          clientFd = accept(sockFd, NULL, NULL);
          if (clientFd >= 0) {
            int flags = fcntl(clientFd, F_GETFL, 0);
            fcntl(clientFd, F_SETFL, flags | O_NONBLOCK);
            std::cout << "[IPC] Cliente conectado." << std::endl;
          }
        }

        // Leer comando del cliente conectado (1 byte)
        if (clientFd >= 0) {
          uint8_t cmdByte;
          ssize_t n = recv(clientFd, &cmdByte, 1, MSG_DONTWAIT);
          if (n == 1) {
            processIpcCommand(cmdByte, effectEnabled);
          } else if (n == 0) {
            close(clientFd);
            clientFd = -1;
            std::cout << "[IPC] Cliente desconectado. Esperando reconexión..."
                      << std::endl;
          }
        }
      }

      cap.read(frame);
      if (frame.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }

      auto start = std::chrono::high_resolution_clock::now();

      // --- Pipeline de visión (solo si efecto habilitado) ---
      cv::Rect roi;
      bool faceFound = false;

      if (effectEnabled && faceDetectorOk && faceSession) {
        cv::resize(frame, ufResized, cv::Size(UF_W, UF_H));
        const int ufPlane = UF_H * UF_W;
        float *ch0 = ufTensorData.data();
        float *ch1 = ch0 + ufPlane;
        float *ch2 = ch1 + ufPlane;

        for (int y = 0; y < UF_H; ++y) {
          const uint8_t *row = ufResized.ptr<uint8_t>(y);
          for (int x = 0; x < UF_W; ++x) {
            int idx = y * UF_W + x;
            int px3 = x * 3;
            ch0[idx] = (row[px3 + 0] - 127.0f) / 128.0f;
            ch1[idx] = (row[px3 + 1] - 127.0f) / 128.0f;
            ch2[idx] = (row[px3 + 2] - 127.0f) / 128.0f;
          }
        }

        try {
          auto ufResults =
              faceSession->Run(runOpts, &ufInputName, &ufInputTensor, 1,
                               ufOutputNames.data(), ufOutputNames.size());

          const float *scores = ufResults[0].GetTensorData<float>();
          const float *boxes = ufResults[1].GetTensorData<float>();
          int numAnchors = static_cast<int>(ufPriors.size());

          float bestConf = 0.7f;
          int bestIdx = -1;

          for (int a = 0; a < numAnchors; ++a) {
            float conf = scores[a * 2 + 1];
            if (conf > bestConf) {
              bestConf = conf;
              bestIdx = a;
            }
          }

          if (bestIdx >= 0) {
            const float CENTER_VAR = 0.1f;
            const float SIZE_VAR = 0.2f;
            float pcx = ufPriors[bestIdx][0];
            float pcy = ufPriors[bestIdx][1];
            float pw = ufPriors[bestIdx][2];
            float ph = ufPriors[bestIdx][3];

            float cx = pcx + boxes[bestIdx * 4 + 0] * CENTER_VAR * pw;
            float cy = pcy + boxes[bestIdx * 4 + 1] * CENTER_VAR * ph;
            float bw = pw * std::exp(boxes[bestIdx * 4 + 2] * SIZE_VAR);
            float bh = ph * std::exp(boxes[bestIdx * 4 + 3] * SIZE_VAR);

            float bx1 = cx - bw / 2.0f;
            float by1 = cy - bh / 2.0f;
            float bx2 = cx + bw / 2.0f;
            float by2 = cy + bh / 2.0f;

            int x1 = static_cast<int>(bx1 * frame.cols);
            int y1 = static_cast<int>(by1 * frame.rows);
            int x2 = static_cast<int>(bx2 * frame.cols);
            int y2 = static_cast<int>(by2 * frame.rows);

            int w = x2 - x1, h = y2 - y1;
            if (w > 10 && h > 10) {
              int padW = w / 10, padH = h / 10;
              x1 = std::max(0, x1 - padW);
              y1 = std::max(0, y1 - padH);
              x2 = std::min(frame.cols, x2 + padW);
              y2 = std::min(frame.rows, y2 + padH);
              roi = cv::Rect(x1, y1, x2 - x1, y2 - y1);
              faceFound = true;
            }
          }
        } catch (const Ort::Exception &e) {
          std::cerr << "[FACE] Error UltraFace: " << e.what() << std::endl;
        }
      }

      if (faceFound) {
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        cv::Mat faceCrop = frame(roi);
        preprocessFrame(faceCrop, resized, tensorData.data());

        if (modelLoaded && session) {
          try {
            auto results = session->Run(runOpts, &cachedInputName, &inputTensor,
                                        1, &cachedOutputName, 1);

            const float *landmarks = results[0].GetTensorData<float>();

            for (int p = 0; p < 98; ++p) {
              float pfld_x = landmarks[p * 2];
              float pfld_y = landmarks[p * 2 + 1];

              int final_x = roi.x + static_cast<int>(pfld_x * roi.width);
              int final_y = roi.y + static_cast<int>(pfld_y * roi.height);

              bool isEyeOrPupil = (p >= 60 && p <= 75) || (p >= 96 && p <= 97);
              cv::Scalar color =
                  isEyeOrPupil ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
              int radius = isEyeOrPupil ? 3 : 2;

              cv::circle(frame, cv::Point(final_x, final_y), radius, color, -1,
                         cv::LINE_AA);
            }

          } catch (const Ort::Exception &e) {
            std::cerr << "[ONNX] Error inferencia: " << e.what() << std::endl;
          }
        }
      } else if (effectEnabled) {
        cv::putText(frame, "ESTADO: BUSCANDO ROSTRO...", cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2,
                    cv::LINE_AA);
      }

      // Escritura a cámara virtual v4l2loopback
      write(fd, frame.data, frame.total() * frame.elemSize());

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> elapsed = end - start;
      total_latency += elapsed.count();
      frames_processed++;

      // Reporte periódico cada 300 frames
      if (frames_processed % 300 == 0) {
        std::cout << "[PIPELINE] Frames: " << frames_processed
                  << " | Latencia promedio: "
                  << (total_latency / frames_processed) << " ms"
                  << " | RAM: " << (getProcessMemory() / (1024.0 * 1024.0))
                  << " MB"
                  << " | Efecto: " << (effectEnabled ? "ON" : "OFF")
                  << std::endl;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Excepcion capturada: " << e.what() << std::endl;
  }

  // -----------------------------------------------------------------
  // Shutdown: higiene de descriptores y desvinculado de sockets
  // -----------------------------------------------------------------
  std::cout << "\n[DAEMON] Iniciando shutdown limpio..." << std::endl;

  if (clientFd >= 0)
    close(clientFd);
  if (sockFd >= 0) {
    close(sockFd);
    unlink(DIRECTLOOK_SOCK_PATH);
    std::cout << "[IPC] Socket UNIX cerrado y desvinculado." << std::endl;
  }

  if (frames_processed > 0) {
    std::cout << "[RESULTADO] Frames totales: " << frames_processed
              << std::endl;
    std::cout << "[RESULTADO] Latencia promedio: "
              << (total_latency / frames_processed) << " ms" << std::endl;
  }

  size_t memFinal = getProcessMemory();
  std::cout << "[MEMORIA] Fin del pipeline: " << (memFinal / (1024.0 * 1024.0))
            << " MB" << std::endl;

  if (fd >= 0)
    close(fd);
  if (cap.isOpened())
    cap.release();
  std::cout << "[DAEMON] Shutdown limpio completado." << std::endl;
  return 0;
}
#endif