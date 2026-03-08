#include "vision_pipeline.h"
#include <iostream>

VisionPipeline::VisionPipeline(const std::string &faceModelPath,
                               const std::string &modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon"),
      ufTensorData(UF_ELEMENTS), tensorData(TENSOR_ELEMENTS) {

  sessionOpts.SetIntraOpNumThreads(1);
  sessionOpts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // --- Face Model (UltraFace) ---
  try {
#ifdef _WIN32
    std::wstring fwpath(faceModelPath.begin(), faceModelPath.end());
    faceSession =
        std::make_unique<Ort::Session>(env, fwpath.c_str(), sessionOpts);
#else
    faceSession =
        std::make_unique<Ort::Session>(env, faceModelPath.c_str(), sessionOpts);
#endif
    faceDetectorOk = true;
    std::cout << "[FACE] UltraFace cargado: " << faceModelPath << std::endl;

    std::array<int64_t, 4> ufShape = {1, 3, UF_H, UF_W};
    auto ufMemInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    ufInputTensor = Ort::Value::CreateTensor<float>(
        ufMemInfo, ufTensorData.data(), ufTensorData.size(), ufShape.data(),
        ufShape.size());

    ufPriors = generateUltraFacePriors(UF_W, UF_H);

    Ort::AllocatorWithDefaultOptions ufAlloc;
    auto ufInNameAllocated = faceSession->GetInputNameAllocated(0, ufAlloc);
    ufInputNameStr = ufInNameAllocated.get();
    ufInputName = ufInputNameStr.c_str();

    for (size_t i = 0; i < faceSession->GetOutputCount(); ++i) {
      auto outNameAllocated = faceSession->GetOutputNameAllocated(i, ufAlloc);
      ufOutNamesStr.push_back(outNameAllocated.get());
      ufOutputNames.push_back(ufOutNamesStr.back().c_str());
    }
  } catch (const Ort::Exception &e) {
    std::cerr << "[FACE] No se pudo cargar UltraFace: " << e.what()
              << std::endl;
  }

  // --- Landmarks Model (PFLD) ---
  try {
#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session = std::make_unique<Ort::Session>(env, wpath.c_str(), sessionOpts);
#else
    session =
        std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOpts);
#endif
    modelLoaded = true;
    std::cout << "[ONNX] Modelo cargado: " << modelPath << std::endl;

    std::array<int64_t, 4> inputShape = {1, 3, MODEL_SIZE, MODEL_SIZE};
    auto memInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(), inputShape.data(),
        inputShape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto inNameAllocated = session->GetInputNameAllocated(0, alloc);
    cachedInputNameStr = inNameAllocated.get();
    cachedInputName = cachedInputNameStr.c_str();

    auto outNameAllocated = session->GetOutputNameAllocated(0, alloc);
    cachedOutputNameStr = outNameAllocated.get();
    cachedOutputName = cachedOutputNameStr.c_str();

  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX] No se pudo cargar modelo PFLD: " << e.what()
              << std::endl;
  }
}

void VisionPipeline::preprocessFrame(const cv::Mat &frame, cv::Mat &outResized,
                                     cv::Mat &outBlobBuffer, float *buffer) {
  cv::resize(frame, outResized, cv::Size(MODEL_SIZE, MODEL_SIZE));
  cv::dnn::blobFromImage(outResized, outBlobBuffer, 1.0 / 255.0, cv::Size(),
                         cv::Scalar(), true, false);
  std::memcpy(buffer, outBlobBuffer.ptr<float>(),
              outBlobBuffer.total() * sizeof(float));
}

std::vector<std::array<float, 4>>
VisionPipeline::generateUltraFacePriors(int imgW, int imgH) {
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

void VisionPipeline::process(cv::Mat &frame, bool effectEnabled) {
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
      std::cerr << "[Inferencia UltraFace Error] " << e.what() << std::endl;
      return;
    }
  }

  if (faceFound) {
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    cv::Mat faceCrop = frame(roi);
    preprocessFrame(faceCrop, resized, blobBuffer, tensorData.data());

    if (modelLoaded && session) {
      try {
        auto results = session->Run(runOpts, &cachedInputName, &inputTensor, 1,
                                    &cachedOutputName, 1);

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
        std::cerr << "[Inferencia PFLD Error] " << e.what() << std::endl;
        return;
      }
    }
  } else if (effectEnabled) {
    cv::putText(frame, "ESTADO: BUSCANDO ROSTRO...", cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2,
                cv::LINE_AA);
  }
}
