// vision_pipeline.cpp
#include "vision_pipeline.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

VisionPipeline::VisionPipeline(const std::string &faceModelPath,
                               const std::string &modelPath, double fps)
    : env(ORT_LOGGING_LEVEL_WARNING, "DirectLookPipeline"),
      faceInputTensorValues(3 * 240 * 320, 0.0f),
      pfldInputTensorValues(3 * 112 * 112, 0.0f) {

  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  try {
#ifdef _WIN32
    std::wstring wFacePath(faceModelPath.begin(), faceModelPath.end());
    std::wstring wPfldPath(modelPath.begin(), modelPath.end());
    faceSession =
        std::make_unique<Ort::Session>(env, wFacePath.c_str(), sessionOptions);
    session =
        std::make_unique<Ort::Session>(env, wPfldPath.c_str(), sessionOptions);
#else
    faceSession = std::make_unique<Ort::Session>(env, faceModelPath.c_str(),
                                                 sessionOptions);
    session =
        std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions alloc;

    auto faceInNameAllocated = faceSession->GetInputNameAllocated(0, alloc);
    faceInputNameStr = faceInNameAllocated.get();
    faceInputName = faceInputNameStr.c_str();

    size_t faceNumOutputs = faceSession->GetOutputCount();
    faceOutputNamesStr.reserve(faceNumOutputs);
    for (size_t i = 0; i < faceNumOutputs; i++) {
      auto outNameAllocated = faceSession->GetOutputNameAllocated(i, alloc);
      faceOutputNamesStr.push_back(outNameAllocated.get());
    }
    for (size_t i = 0; i < faceNumOutputs; i++) {
      faceOutputNames.push_back(faceOutputNamesStr[i].c_str());
    }

    auto pfldInNameAllocated = session->GetInputNameAllocated(0, alloc);
    pfldInputNameStr = pfldInNameAllocated.get();
    pfldInputName = pfldInputNameStr.c_str();

    auto pfldOutNameAllocated = session->GetOutputNameAllocated(0, alloc);
    pfldOutputNameStr = pfldOutNameAllocated.get();
    pfldOutputName = pfldOutputNameStr.c_str();

    generatePriors(320, 240);

  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX] Falla en inicialización Pipeline: " << e.what()
              << std::endl;
    throw;
  }
}

VisionPipeline::~VisionPipeline() {}

void VisionPipeline::generatePriors(int img_w, int img_h) {
  priors.clear();
  std::vector<int> strides = {8, 16, 32, 64};
  std::vector<std::vector<float>> min_boxes = {{10.0f, 16.0f, 24.0f},
                                               {32.0f, 48.0f},
                                               {64.0f, 96.0f},
                                               {128.0f, 192.0f, 256.0f}};

  for (int i = 0; i < 4; ++i) {
    int fm_h = std::ceil((float)img_h / strides[i]);
    int fm_w = std::ceil((float)img_w / strides[i]);
    for (int y = 0; y < fm_h; ++y) {
      for (int x = 0; x < fm_w; ++x) {
        for (float mb : min_boxes[i]) {
          float cx = (x + 0.5f) * strides[i] / img_w;
          float cy = (y + 0.5f) * strides[i] / img_h;
          float w = mb / img_w;
          float h = mb / img_h;
          priors.push_back({cx, cy, w, h});
        }
      }
    }
  }
}

bool VisionPipeline::hasValidEyes() const { return validEyes; }

cv::Rect VisionPipeline::getLeftEyeRoi() const { return currentLeftRoi; }

cv::Rect VisionPipeline::getRightEyeRoi() const { return currentRightRoi; }

void VisionPipeline::process(cv::Mat &frame, bool effectEnabled,
                             int degradationLevel) {
  validEyes = false;
  currentLeftRoi = cv::Rect();
  currentRightRoi = cv::Rect();

  if (!effectEnabled || degradationLevel >= 3 || frame.empty())
    return;

  int w_f = frame.cols;
  int h_f = frame.rows;

  cv::Mat resized_face;
  cv::resize(frame, resized_face, cv::Size(320, 240));

  cv::Mat rgb_face;
  cv::cvtColor(resized_face, rgb_face, cv::COLOR_BGR2RGB);

  cv::Mat rgb_float;
  rgb_face.convertTo(rgb_float, CV_32FC3, 1.0f / 128.0f, -127.0f / 128.0f);

  cv::Mat face_channels[3];
  for (int c = 0; c < 3; ++c) {
    face_channels[c] = cv::Mat(240, 320, CV_32FC1,
                               faceInputTensorValues.data() + c * 240 * 320);
  }
  cv::split(rgb_float, face_channels);

  std::array<int64_t, 4> faceInputShape = {1, 3, 240, 320};
  auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value faceInputTensor = Ort::Value::CreateTensor<float>(
      memInfo, faceInputTensorValues.data(), faceInputTensorValues.size(),
      faceInputShape.data(), faceInputShape.size());

  const char *faceInputNames[] = {faceInputName};

  try {
    auto faceOutputTensors = faceSession->Run(
        Ort::RunOptions{nullptr}, faceInputNames, &faceInputTensor, 1,
        faceOutputNames.data(), faceOutputNames.size());

    const float *scores = faceOutputTensors[0].GetTensorData<float>();
    const float *boxes = faceOutputTensors[1].GetTensorData<float>();

    int num_priors = priors.size();
    float best_score = 0.0f;
    int best_idx = -1;

    for (int i = 0; i < num_priors; ++i) {
      float score = scores[i * 2 + 1];
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }

    if (best_score < 0.7f || best_idx == -1)
      return;

    float pcx = priors[best_idx][0];
    float pcy = priors[best_idx][1];
    float pw = priors[best_idx][2];
    float ph = priors[best_idx][3];

    float bx = boxes[best_idx * 4 + 0];
    float by = boxes[best_idx * 4 + 1];
    float bw = boxes[best_idx * 4 + 2];
    float bh = boxes[best_idx * 4 + 3];

    float cx = pcx + bx * 0.1f * pw;
    float cy = pcy + by * 0.1f * ph;
    float w = pw * std::exp(bw * 0.2f);
    float h = ph * std::exp(bh * 0.2f);

    int face_x1 = static_cast<int>((cx - w / 2.0f) * w_f);
    int face_y1 = static_cast<int>((cy - h / 2.0f) * h_f);
    int face_x2 = static_cast<int>((cx + w / 2.0f) * w_f);
    int face_y2 = static_cast<int>((cy + h / 2.0f) * h_f);

    int pad_w = (face_x2 - face_x1) / 10;
    int pad_h = (face_y2 - face_y1) / 10;

    face_x1 = std::max(0, face_x1 - pad_w);
    face_y1 = std::max(0, face_y1 - pad_h);
    face_x2 = std::min(w_f, face_x2 + pad_w);
    face_y2 = std::min(h_f, face_y2 + pad_h);

    int face_w = face_x2 - face_x1;
    int face_h = face_y2 - face_y1;

    if (face_w < 10 || face_h < 10)
      return;

    cv::Mat face_crop = frame(cv::Rect(face_x1, face_y1, face_w, face_h));

    cv::Mat pfld_resized;
    cv::resize(face_crop, pfld_resized, cv::Size(112, 112));

    cv::Mat pfld_rgb;
    cv::cvtColor(pfld_resized, pfld_rgb, cv::COLOR_BGR2RGB);

    cv::Mat pfld_float;
    pfld_rgb.convertTo(pfld_float, CV_32FC3, 1.0f / 255.0f);

    cv::Mat pfld_channels[3];
    for (int c = 0; c < 3; ++c) {
      pfld_channels[c] = cv::Mat(112, 112, CV_32FC1,
                                 pfldInputTensorValues.data() + c * 112 * 112);
    }
    cv::split(pfld_float, pfld_channels);

    std::array<int64_t, 4> pfldInputShape = {1, 3, 112, 112};
    Ort::Value pfldInputTensor = Ort::Value::CreateTensor<float>(
        memInfo, pfldInputTensorValues.data(), pfldInputTensorValues.size(),
        pfldInputShape.data(), pfldInputShape.size());

    const char *pfldInputNames[] = {pfldInputName};
    const char *pfldOutputNames[] = {pfldOutputName};

    auto pfldOutputTensors =
        session->Run(Ort::RunOptions{nullptr}, pfldInputNames, &pfldInputTensor,
                     1, pfldOutputNames, 1);

    const float *landmarks = pfldOutputTensors[0].GetTensorData<float>();

    auto create64x64Roi = [&](int start_idx, int end_idx) -> cv::Rect {
      float min_x = 1e9f, max_x = -1e9f;
      float min_y = 1e9f, max_y = -1e9f;

      for (int i = start_idx; i <= end_idx; ++i) {
        float lx = landmarks[i * 2 + 0];
        float ly = landmarks[i * 2 + 1];

        float abs_x = face_x1 + lx * face_w;
        float abs_y = face_y1 + ly * face_h;

        if (abs_x < min_x)
          min_x = abs_x;
        if (abs_x > max_x)
          max_x = abs_x;
        if (abs_y < min_y)
          min_y = abs_y;
        if (abs_y > max_y)
          max_y = abs_y;
      }

      float ew = max_x - min_x;
      float eh = max_y - min_y;

      float pad_x = ew * 0.2f;
      float pad_y = eh * 0.2f;

      float abs_x1 = min_x - pad_x;
      float abs_y1 = min_y - pad_y;
      float abs_x2 = max_x + pad_x;
      float abs_y2 = max_y + pad_y;

      int cx = static_cast<int>((abs_x1 + abs_x2) / 2);
      int cy = static_cast<int>((abs_y1 + abs_y2) / 2);

      int final_x1 = cx - 32;
      int final_y1 = cy - 32;

      if (final_x1 < 0)
        final_x1 = 0;
      if (final_y1 < 0)
        final_y1 = 0;
      if (final_x1 + 64 > w_f)
        final_x1 = w_f - 64;
      if (final_y1 + 64 > h_f)
        final_y1 = h_f - 64;

      if (w_f >= 64 && h_f >= 64) {
        return cv::Rect(final_x1, final_y1, 64, 64);
      }
      return cv::Rect();
    };

    currentLeftRoi = create64x64Roi(36, 41);
    currentRightRoi = create64x64Roi(42, 47);

    if (currentLeftRoi.area() > 0 && currentRightRoi.area() > 0) {
      validEyes = true;
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "[Pipeline] Hardware ONNX abortado geométricamente: "
              << e.what() << std::endl;
    validEyes = false;
  }
}
