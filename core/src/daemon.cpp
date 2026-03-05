// =============================================================================
// DirectLook — Daemon Core (Multiplataforma: Windows + Linux)
// Sprint 1: Monitor de Memoria · ONNX Runtime · Transmutación Tensorial
// =============================================================================
// --- Includes comunes (agnósticos de plataforma) ---
#include <chrono>
#include <cstdint>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// --- Includes condicionales de plataforma ---

#ifdef _WIN32
#include <windows.h>

#include <psapi.h>
#else

#include <fcntl.h>
#include <fstream>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#endif

// =============================================================================
// Constantes arquitectónicas
// =============================================================================
static constexpr size_t MEMORY_LIMIT_BYTES = 80 * 1024 * 1024; // 80 MB
static const std::string MODEL_PATH = "modelos/directlook_int8.onnx";

// Dimensiones exigidas por el modelo MobileNet V2
static constexpr int MODEL_SIZE = 448;
static constexpr int TENSOR_ELEMENTS = 1 * 3 * MODEL_SIZE * MODEL_SIZE;

// =============================================================================
// [FASE 1] Monitor de Memoria Multiplataforma
// =============================================================================

size_t getProcessMemory() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return static_cast<size_t>(pmc.WorkingSetSize);
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
// [FASE 3] Transmutación de Tensores: BGR/HWC → RGB/NCHW float [0.0, 1.0]
// Resize a 448x448, conversión de color, transposición y normalización
// en un solo pase con punteros directos. El buffer y el cv::Mat de resize
// se reciben pre-alocados para evitar allocations por frame.
// =============================================================================
void preprocessFrame(const cv::Mat &frame, cv::Mat &resized, float *buffer) {
  // 1. Resize forzado a las dimensiones del modelo (reutiliza memoria de
  // resized)
  cv::resize(frame, resized, cv::Size(MODEL_SIZE, MODEL_SIZE));
  const int planeSize = MODEL_SIZE * MODEL_SIZE;
  // Punteros directos a los planos R, G, B en layout NCHW
  float *rPlane = buffer;
  float *gPlane = buffer + planeSize;
  float *bPlane = buffer + 2 * planeSize;
  constexpr float INV_255 = 1.0f / 255.0f;
  for (int y = 0; y < MODEL_SIZE; ++y) {
    const uint8_t *row = resized.ptr<uint8_t>(y);
    const int rowOffset = y * MODEL_SIZE;
    for (int x = 0; x < MODEL_SIZE; ++x) {
      const int px = x * 3;
      const int idx = rowOffset + x;
      // BGR (OpenCV) → RGB (ONNX) + normalización en un solo paso
      rPlane[idx] = row[px + 2] * INV_255; // R
      gPlane[idx] = row[px + 1] * INV_255; // G
      bPlane[idx] = row[px + 0] * INV_255; // B
    }
  }
}

// =============================================================================
// Arquitecturas de plataforma (main separados)
// =============================================================================

#ifdef _WIN32

// =====================================================================
// ARQUITECTURA WINDOWS (DirectShow / MSMF)
// =====================================================================

int main() {
  std::cout << "=== DirectLook Daemon [Windows] ===" << std::endl;
  // Reporte de memoria pre-inicialización
  size_t memInicio = getProcessMemory();
  std::cout << "[MEMORIA] Inicio del proceso: "
            << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;
  // -----------------------------------------------------------------
  // [FASE 2] Instanciación del Motor ONNX Runtime (CPU / AVX2)
  // -----------------------------------------------------------------
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");
  Ort::SessionOptions sessionOpts;

  sessionOpts.SetIntraOpNumThreads(2);
  sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  sessionOpts.DisableCpuMemArena(); // Desactiva pre-reserva de RAM (~80MB)
  sessionOpts.DisableMemPattern();  // Desactiva patrón de memoria agresivo

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

  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir cámara en Windows."
              << std::endl;
    return 1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  cap.set(cv::CAP_PROP_FPS, 15);

  // -----------------------------------------------------------------
  // Loop de benchmark + preprocesamiento tensorial + inferencia
  // -----------------------------------------------------------------

  cv::Mat frame;
  cv::Mat resized;
  const int benchmark_frames = 600;
  double total_latency = 0.0;
  int frames_processed = 0;

  // Buffers pre-alocados FUERA del bucle
  std::vector<float> tensorData(TENSOR_ELEMENTS);
  std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
  auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Nombres cacheados
  const char *cachedInputName = nullptr;
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::AllocatedStringPtr inNameHolder(nullptr,
                                       Ort::detail::AllocatedFree{nullptr});
  Ort::RunOptions runOpts;

  // Tensor de entrada creado UNA vez (apunta a tensorData, zero-copy)
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
      inputShape.size());

  // Tensores de salida pre-alocados para TODAS las salidas del modelo
  std::vector<std::vector<float>> outputBuffers;
  std::vector<Ort::Value> outputTensors;
  std::vector<Ort::AllocatedStringPtr> outNameHolders;
  std::unique_ptr<Ort::IoBinding> ioBinding;

  if (modelLoaded && session) {
    inNameHolder = session->GetInputNameAllocated(0, alloc);
    cachedInputName = inNameHolder.get();

    size_t numOutputs = session->GetOutputCount();
    std::cout << "[ONNX] Vinculando " << numOutputs << " salida(s)..."
              << std::endl;

    ioBinding = std::make_unique<Ort::IoBinding>(*session);
    ioBinding->BindInput(cachedInputName, inputTensor);

    for (size_t i = 0; i < numOutputs; ++i) {
      auto nameHolder = session->GetOutputNameAllocated(i, alloc);
      auto outInfo = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto outShape = outInfo.GetShape();
      size_t outSize = 1;
      for (auto &dim : outShape)
        outSize *= (dim < 0) ? 1 : static_cast<size_t>(dim);

      outputBuffers.emplace_back(outSize);
      outputTensors.push_back(Ort::Value::CreateTensor<float>(
          memInfo, outputBuffers.back().data(), outputBuffers.back().size(),
          outShape.data(), outShape.size()));

      ioBinding->BindOutput(nameHolder.get(), outputTensors.back());
      std::cout << "[ONNX]   Bound output[" << i << "]: '" << nameHolder.get()
                << "' size=" << outSize << std::endl;
      outNameHolders.push_back(std::move(nameHolder));
    }
  }

  std::cout << "[BENCHMARK] Iniciando captura (" << benchmark_frames
            << " frames)..." << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    cap.read(frame);
    if (frame.empty())
      break;
    auto start = std::chrono::high_resolution_clock::now();

    // [FASE 3] Transmutación tensorial (resize 448x448 + BGR→RGB + NCHW)
    // Escribe directamente en tensorData.data() que el inputTensor ya apunta
    preprocessFrame(frame, resized, tensorData.data());

    // [FASE 2] Inferencia ONNX zero-copy via IoBinding
    if (modelLoaded && session && ioBinding) {
      try {
        session->Run(runOpts, *ioBinding);
        // Resultados escritos directamente en outputValues[]
      } catch (const Ort::Exception &e) {
        std::cerr << "[ONNX] Error inferencia frame " << i << ": " << e.what()
                  << std::endl;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    total_latency += elapsed.count();
    frames_processed++;

    if ((i + 1) % 100 == 0) {
      std::cout << "[PIPELINE] Frame " << (i + 1) << "/" << benchmark_frames
                << " | Latencia promedio: "
                << (total_latency / frames_processed) << " ms"
                << " | RAM: " << (getProcessMemory() / (1024.0 * 1024.0))
                << " MB" << std::endl;
    }
  }

  // -----------------------------------------------------------------
  // Reporte final
  // -----------------------------------------------------------------

  std::cout << std::endl;
  if (frames_processed > 0) {
    std::cout << "[RESULTADO] Frames: " << frames_processed << std::endl;
    std::cout << "[RESULTADO] Latencia promedio: "
              << (total_latency / frames_processed) << " ms" << std::endl;
  }

  size_t memFinal = getProcessMemory();
  std::cout << "[MEMORIA] Fin del pipeline: " << (memFinal / (1024.0 * 1024.0))
            << " MB" << std::endl;

  cap.release();
  std::cout << "[DAEMON] Shutdown limpio." << std::endl;
  return 0;
}

#else
// =====================================================================
// ARQUITECTURA LINUX (v4l2loopback nativo)
// =====================================================================

int main() {
  std::cout << "=== DirectLook Daemon [Linux] ===" << std::endl;

  // Reporte de memoria pre-inicialización
  size_t memInicio = getProcessMemory();
  std::cout << "[MEMORIA] Inicio del proceso: "
            << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

  // -----------------------------------------------------------------
  // [FASE 2] Instanciación del Motor ONNX Runtime (CPU / AVX2)
  // -----------------------------------------------------------------
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");

  Ort::SessionOptions sessionOpts;
  sessionOpts.SetIntraOpNumThreads(2);
  sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  sessionOpts.DisableCpuMemArena();
  sessionOpts.DisableMemPattern();

  std::unique_ptr<Ort::Session> session;
  bool modelLoaded = false;

  try {
    session =
        std::make_unique<Ort::Session>(env, MODEL_PATH.c_str(), sessionOpts);
    modelLoaded = true;
    std::cout << "[ONNX] Modelo cargado: " << MODEL_PATH << std::endl;

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
  // Captura de video + cámara virtual
  // -----------------------------------------------------------------
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir /dev/video0."
              << std::endl;
    return 1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  cap.set(cv::CAP_PROP_FPS, 15);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  int fd = open("/dev/video2", O_RDWR);
  if (fd < 0) {
    std::cerr << "Falla de kernel: Descriptor /dev/video2 inaccesible."
              << std::endl;
    return 1;
  }

  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = 640;
  fmt.fmt.pix.height = 360;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
  fmt.fmt.pix.sizeimage = 640 * 360 * 3;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    std::cerr
        << "Falla de I/O: Imposible negociar formato en descriptor virtual."
        << std::endl;
    return 1;
  }

  // -----------------------------------------------------------------
  // Loop de benchmark + preprocesamiento tensorial + inferencia
  // -----------------------------------------------------------------
  cv::Mat frame;
  cv::Mat resized;
  const int benchmark_frames = 600;
  double total_latency = 0.0;
  int frames_processed = 0;

  // Buffers pre-alocados FUERA del bucle
  std::vector<float> tensorData(TENSOR_ELEMENTS);
  std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
  auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Nombres cacheados
  const char *cachedInputName = nullptr;
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::AllocatedStringPtr inNameHolder(nullptr,
                                       Ort::detail::AllocatedFree{nullptr});
  Ort::RunOptions runOpts;

  // Tensor de entrada creado UNA vez (zero-copy sobre tensorData)
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
      inputShape.size());

  // Tensores de salida pre-alocados para TODAS las salidas
  std::vector<std::vector<float>> outputBuffers;
  std::vector<Ort::Value> outputTensors;
  std::vector<Ort::AllocatedStringPtr> outNameHolders;
  std::unique_ptr<Ort::IoBinding> ioBinding;

  if (modelLoaded && session) {
    inNameHolder = session->GetInputNameAllocated(0, alloc);
    cachedInputName = inNameHolder.get();

    size_t numOutputs = session->GetOutputCount();
    ioBinding = std::make_unique<Ort::IoBinding>(*session);
    ioBinding->BindInput(cachedInputName, inputTensor);

    for (size_t i = 0; i < numOutputs; ++i) {
      auto nameHolder = session->GetOutputNameAllocated(i, alloc);
      auto outInfo = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto outShape = outInfo.GetShape();
      size_t outSize = 1;
      for (auto &dim : outShape)
        outSize *= (dim < 0) ? 1 : static_cast<size_t>(dim);

      outputBuffers.emplace_back(outSize);
      outputTensors.push_back(Ort::Value::CreateTensor<float>(
          memInfo, outputBuffers.back().data(), outputBuffers.back().size(),
          outShape.data(), outShape.size()));

      ioBinding->BindOutput(nameHolder.get(), outputTensors.back());
      outNameHolders.push_back(std::move(nameHolder));
    }
  }

  std::cout << "[BENCHMARK] Productor activo. Ejecuta 'mpv' en la segunda "
               "terminal AHORA."
            << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    cap.read(frame);
    if (frame.empty())
      break;

    auto start = std::chrono::high_resolution_clock::now();

    // [FASE 3] Transmutación tensorial (resize 448x448 + BGR→RGB + NCHW)
    preprocessFrame(frame, resized, tensorData.data());

    // [FASE 2] Inferencia ONNX zero-copy via IoBinding
    if (modelLoaded && session && ioBinding) {
      try {
        session->Run(runOpts, *ioBinding);
      } catch (const Ort::Exception &e) {
        std::cerr << "[ONNX] Error inferencia frame " << i << ": " << e.what()
                  << std::endl;
      }
    }

    // Escritura a cámara virtual v4l2loopback
    write(fd, frame.data, frame.total() * frame.elemSize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    total_latency += elapsed.count();
    frames_processed++;

    if ((i + 1) % 100 == 0) {
      std::cout << "[PIPELINE] Frame " << (i + 1) << "/" << benchmark_frames
                << " | Latencia promedio: "
                << (total_latency / frames_processed) << " ms"
                << " | RAM: " << (getProcessMemory() / (1024.0 * 1024.0))
                << " MB" << std::endl;
    }
  }

  // -----------------------------------------------------------------
  // Reporte final
  // -----------------------------------------------------------------
  std::cout << std::endl;
  if (frames_processed > 0) {
    std::cout << "[RESULTADO] Frames: " << frames_processed << std::endl;
    std::cout << "[RESULTADO] Latencia promedio: "
              << (total_latency / frames_processed) << " ms" << std::endl;
  }

  size_t memFinal = getProcessMemory();
  std::cout << "[MEMORIA] Fin del pipeline: " << (memFinal / (1024.0 * 1024.0))
            << " MB" << std::endl;

  close(fd);
  cap.release();
  std::cout << "[DAEMON] Shutdown limpio." << std::endl;
  return 0;
}
#endif