#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

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
#include <unistd.h>

#endif

// Constantes arquitectónicas
static constexpr size_t MEMORY_LIMIT_BYTES = 80 * 1024 * 1024; // 80 MB
static const std::string MODEL_PATH = "modelos/pfld.onnx";
static const std::string FACE_MODEL_PATH =
    "modelos/version-slim-320_simplified.onnx";

// Dimensiones exigidas por el modelo PFLD (112x112)
static constexpr int MODEL_SIZE = 112;
static constexpr int TENSOR_ELEMENTS = 1 * 3 * MODEL_SIZE * MODEL_SIZE;

// [FASE 1] Monitor de Memoria Multiplataforma
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

// [FASE 3] Transmutación de Tensores: BGR/HWC → RGB/NCHW float [0.0, 1.0]

void preprocessFrame(const cv::Mat &frame, cv::Mat &resized, float *buffer) {
  cv::resize(frame, resized, cv::Size(MODEL_SIZE, MODEL_SIZE));
  cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(),
                                        cv::Scalar(), true);
  std::memcpy(buffer, blob.ptr<float>(), blob.total() * sizeof(float));
}

// Generador de Prior Boxes para UltraFace SSD (slim-320)
// Produce 4420 anchors basados en feature maps [30x40, 15x20, 8x10, 4x5]

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

// Arquitecturas de plataforma (main separados)

#ifdef _WIN32

// ARQUITECTURA WINDOWS (DirectShow / MSMF)

int main() {
  std::cout << "=== DirectLook Daemon [Windows] ===" << std::endl;
  // Reporte de memoria pre-inicialización
  size_t memInicio = getProcessMemory();
  std::cout << "[MEMORIA] Inicio del proceso: "
            << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");
  Ort::SessionOptions sessionOpts;

  sessionOpts.SetIntraOpNumThreads(1);
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

  // Captura de video

  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir cámara en Windows."
              << std::endl;
    return 1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  cap.set(cv::CAP_PROP_FPS, 15);

  // Detector facial UltraFace ONNX (reemplaza Haar cascade)

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
      ufTensorData.data(), ufTensorData.size(), ufShape.data(), ufShape.size());

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

  // Loop de benchmark + preprocesamiento tensorial + inferencia
  cv::Mat frame;
  cv::Mat resized;
  cv::Mat ufResized;
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
  const char *cachedOutputName = nullptr;
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::AllocatedStringPtr inNameHolder(nullptr,
                                       Ort::detail::AllocatedFree{nullptr});
  Ort::AllocatedStringPtr outNameHolder(nullptr,
                                        Ort::detail::AllocatedFree{nullptr});
  Ort::RunOptions runOpts;

  // Tensor de entrada creado UNA vez (apunta a tensorData, zero-copy)
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
      inputShape.size());

  if (modelLoaded && session) {
    inNameHolder = session->GetInputNameAllocated(0, alloc);
    outNameHolder =
        session->GetOutputNameAllocated(0, alloc); // 'linear' [1,196]
    cachedInputName = inNameHolder.get();
    cachedOutputName = outNameHolder.get();
  }

  std::cout << "[BENCHMARK] Iniciando captura (" << benchmark_frames
            << " frames)..." << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    cap.read(frame);
    if (frame.empty())
      break;
    auto start = std::chrono::high_resolution_clock::now();

    // [PASO 1] Detectar rostro con UltraFace ONNX
    cv::Rect roi;
    bool faceFound = false;

    if (faceDetectorOk && faceSession) {
      // Preprocesar para UltraFace: resize 320x240, (px-127)/128, BGR
      cv::resize(frame, ufResized, cv::Size(UF_W, UF_H));
      const int ufPlane = UF_H * UF_W;
      float *ch0 = ufTensorData.data(); // B (canal 0)
      float *ch1 = ch0 + ufPlane;       // G (canal 1)
      float *ch2 = ch1 + ufPlane;       // R (canal 2)
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

        // scores: [1, 4420, 2]  boxes (raw offsets): [1, 4420, 4]
        const float *scores = ufResults[0].GetTensorData<float>();
        const float *boxes = ufResults[1].GetTensorData<float>();
        auto shape0 = ufResults[0].GetTensorTypeAndShapeInfo().GetShape();
        auto shape1 = ufResults[1].GetTensorTypeAndShapeInfo().GetShape();

        // Debug: imprimir shapes en el primer frame
        if (i == 0) {
          std::cout << "[DEBUG] UltraFace Output[0] shape=[";
          for (size_t s = 0; s < shape0.size(); ++s)
            std::cout << shape0[s] << (s < shape0.size() - 1 ? "," : "");
          std::cout << "]" << std::endl;
          std::cout << "[DEBUG] UltraFace Output[1] shape=[";
          for (size_t s = 0; s < shape1.size(); ++s)
            std::cout << shape1[s] << (s < shape1.size() - 1 ? "," : "");
          std::cout << "]" << std::endl;
        }

        int numAnchors = static_cast<int>(ufPriors.size());

        // Encontrar la cara con mayor confianza
        float bestConf = 0.7f; // Umbral mínimo
        int bestIdx = -1;
        for (int a = 0; a < numAnchors; ++a) {
          float conf = scores[a * 2 + 1]; // [bg, face]
          if (conf > bestConf) {
            bestConf = conf;
            bestIdx = a;
          }
        }

        if (bestIdx >= 0) {
          // Decodificar SSD: offset + prior → coordenadas reales
          const float CENTER_VAR = 0.1f;
          const float SIZE_VAR = 0.2f;
          float pcx = ufPriors[bestIdx][0]; // prior center x
          float pcy = ufPriors[bestIdx][1]; // prior center y
          float pw = ufPriors[bestIdx][2];  // prior width
          float ph = ufPriors[bestIdx][3];  // prior height

          float cx = pcx + boxes[bestIdx * 4 + 0] * CENTER_VAR * pw;
          float cy = pcy + boxes[bestIdx * 4 + 1] * CENTER_VAR * ph;
          float bw = pw * std::exp(boxes[bestIdx * 4 + 2] * SIZE_VAR);
          float bh = ph * std::exp(boxes[bestIdx * 4 + 3] * SIZE_VAR);

          // Centro → esquinas (normalizadas [0-1])
          float bx1 = cx - bw / 2.0f;
          float by1 = cy - bh / 2.0f;
          float bx2 = cx + bw / 2.0f;
          float by2 = cy + bh / 2.0f;

          // Debug: imprimir coordenadas del box
          if (i % 100 == 0) {
            std::cout << "[DEBUG] Face box: (" << bx1 << ", " << by1 << ") -> ("
                      << bx2 << ", " << by2 << ") conf=" << bestConf
                      << std::endl;
          }

          int x1 = static_cast<int>(bx1 * frame.cols);
          int y1 = static_cast<int>(by1 * frame.rows);
          int x2 = static_cast<int>(bx2 * frame.cols);
          int y2 = static_cast<int>(by2 * frame.rows);

          // Padding 10% + bounds clamping
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
      // Dibujar rectángulo verde del rostro
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
          std::cerr << "[ONNX] Error inferencia frame " << i << ": " << e.what()
                    << std::endl;
        }
      }
    } else {
      // Sin rostro detectado: no ejecutar PFLD
      cv::putText(frame, "ESTADO: BUSCANDO ROSTRO...", cv::Point(10, 40),
                  cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2,
                  cv::LINE_AA);
    }

    // Vista en vivo de la malla facial
    cv::imshow("DirectLook - PFLD Landmarks", frame);
    if (cv::waitKey(1) == 27)
      break;

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

  // Reporte final
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
  cv::destroyAllWindows();
  std::cout << "[DAEMON] Shutdown limpio." << std::endl;
  return 0;
}

#else

// ARQUITECTURA LINUX (v4l2loopback nativo)

int main() {
  std::cout << "=== DirectLook Daemon [Linux] ===" << std::endl;

  // Reporte de memoria pre-inicialización
  size_t memInicio = getProcessMemory();
  std::cout << "[MEMORIA] Inicio del proceso: "
            << (memInicio / (1024.0 * 1024.0)) << " MB" << std::endl;

  // [FASE 2] Instanciación del Motor ONNX Runtime (CPU / AVX2)
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon");

  Ort::SessionOptions sessionOpts;
  sessionOpts.SetIntraOpNumThreads(1);
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
      std::cerr << "[ADVERTENCIA] Límite arquitectónico de 80MB excedido ("
                << memMB << " MB)" << std::endl;
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX] No se pudo cargar modelo: " << e.what() << std::endl;
    std::cout << "[ONNX] Continuando en modo benchmark (sin inferencia)."
              << std::endl;
  }

  // Captura de video + cámara virtual (/dev/video2)

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

  // CARGA DE MODELO ULTRAFACE ONNX
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

  // Dimensiones UltraFace
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

  // Generar boxes a priori una sola vez
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

  // Buffers PFLD
  cv::Mat frame;
  cv::Mat resized;
  cv::Mat ufResized;
  const int benchmark_frames = 600;
  double total_latency = 0.0;
  int frames_processed = 0;

  // Buffers PFLD pre-alocados
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

  // Tensor PFLD
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
      inputShape.size());

  if (modelLoaded && session) {
    inNameHolder = session->GetInputNameAllocated(0, alloc);
    outNameHolder = session->GetOutputNameAllocated(0, alloc);
    cachedInputName = inNameHolder.get();
    cachedOutputName = outNameHolder.get();
  }

  std::cout << "[BENCHMARK] Productor activo. Ejecuta 'mpv' en la segunda "
               "terminal AHORA."
            << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    cap.read(frame);
    if (frame.empty())
      break;

    auto start = std::chrono::high_resolution_clock::now();

    // [PASO 1] Detectar rostro y extraer ROI con UltraFace
    cv::Rect roi;
    bool faceFound = false;

    if (faceDetectorOk && faceSession) {
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
          std::cerr << "[ONNX] Error inferencia frame " << i << ": " << e.what()
                    << std::endl;
        }
      }
    } else {
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

    if ((i + 1) % 100 == 0) {
      std::cout << "[PIPELINE] Frame " << (i + 1) << "/" << benchmark_frames
                << " | Latencia promedio: "
                << (total_latency / frames_processed) << " ms"
                << " | RAM: " << (getProcessMemory() / (1024.0 * 1024.0))
                << " MB" << std::endl;
    }
  }

  // Reporte final
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