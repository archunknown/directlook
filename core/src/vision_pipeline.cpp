// vision_pipeline.cpp
#include "vision_pipeline.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

VisionPipeline::VisionPipeline(const std::string &faceModelPath,
                               const std::string &modelPath,
                               const std::string &irisModelPath, double fps)
    : env(ORT_LOGGING_LEVEL_WARNING, "DirectLookPipeline"),
      faceInputTensorValues(3 * 240 * 320, 0.0f),
      pfldInputTensorValues(3 * 112 * 112, 0.0f),
      irisInputTensorValues(3 * 64 * 64, 0.0f),
      landmarkFilter_(OneEuroFilterConfig{fps > 0.0 ? fps : 30.0, 1.2, 0.02,
                                          1.0, 0.5}),
      nominalDtSeconds_(fps > 0.0 ? 1.0 / fps : 1.0 / 30.0) {

  sessionOptions.SetIntraOpNumThreads(4);
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

  // --- Iris session (non-fatal: falls back to centre (32,32) if unavailable) ---
  try {
#ifdef _WIN32
    std::wstring wIrisPath(irisModelPath.begin(), irisModelPath.end());
    irisSession = std::make_unique<Ort::Session>(env, wIrisPath.c_str(), sessionOptions);
#else
    irisSession = std::make_unique<Ort::Session>(env, irisModelPath.c_str(), sessionOptions);
#endif
    Ort::AllocatorWithDefaultOptions irisAlloc;
    auto irisInName = irisSession->GetInputNameAllocated(0, irisAlloc);
    irisInputNameStr = irisInName.get();
    irisInputName    = irisInputNameStr.c_str();
    // Find output index for "output_iris" (index 0 on the MediaPipe model).
    size_t irisNumOut = irisSession->GetOutputCount();
    for (size_t i = 0; i < irisNumOut; ++i) {
      auto outName = irisSession->GetOutputNameAllocated(i, irisAlloc);
      std::string name = outName.get();
      if (name == "output_iris") {
        irisOutputNameStr = name;
        irisOutputName    = irisOutputNameStr.c_str();
        break;
      }
    }
    if (!irisOutputName) {
      // Fallback: use first output regardless of name.
      auto out0 = irisSession->GetOutputNameAllocated(0, irisAlloc);
      irisOutputNameStr = out0.get();
      irisOutputName    = irisOutputNameStr.c_str();
    }
    std::cout << "[Iris] Modelo cargado: " << irisModelPath
              << " | input=" << irisInputNameStr
              << " | output=" << irisOutputNameStr << std::endl;
  } catch (const Ort::Exception &e) {
    std::cerr << "[Iris] Fallo al cargar iris_landmark.onnx — usando fallback (32,32): "
              << e.what() << std::endl;
    irisSession.reset();
  }
}

VisionPipeline::~VisionPipeline() {}

// ---------------------------------------------------------------------------
// detectIrisPupil — runs iris_landmark.onnx on a 64×64 BGR crop.
// Returns the iris centre in crop-local coordinates.
// Falls back to (32,32) if the session is null or inference throws.
// ---------------------------------------------------------------------------
cv::Point2f VisionPipeline::detectIrisPupil(const cv::Mat &eyeCrop64) const {
  const cv::Point2f kFallback(32.0f, 32.0f);
  if (!irisSession || eyeCrop64.empty())
    return kFallback;

  // Prepare input: BGR→RGB, float32, normalised to [0,1], CHW layout.
  cv::Mat rgb;
  cv::cvtColor(eyeCrop64, rgb, cv::COLOR_BGR2RGB);

  cv::Mat rgb64;
  if (rgb.cols != 64 || rgb.rows != 64)
    cv::resize(rgb, rgb64, cv::Size(64, 64));
  else
    rgb64 = rgb;

  cv::Mat flt;
  rgb64.convertTo(flt, CV_32FC3, 1.0f / 255.0f);

  // Split into CHW.
  cv::Mat chs[3];
  for (int c = 0; c < 3; ++c)
    chs[c] = cv::Mat(64, 64, CV_32FC1,
                     const_cast<float *>(irisInputTensorValues.data()) + c * 64 * 64);
  cv::split(flt, chs);

  std::array<int64_t, 4> shape = {1, 3, 64, 64};
  auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memInfo,
      const_cast<float *>(irisInputTensorValues.data()),
      irisInputTensorValues.size(),
      shape.data(), shape.size());

  const char *inNames[]  = {irisInputName};
  const char *outNames[] = {irisOutputName};

  try {
    auto outputs = irisSession->Run(
        Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames, 1);
    const float *iris = outputs[0].GetTensorData<float>();
    // output_iris: [1, 15] — landmarks 0..4 each (x,y,z).
    // Index 0 = iris centre x, index 1 = iris centre y, in 64×64 space.
    float cx = std::clamp(iris[0], 0.0f, 63.0f);
    float cy = std::clamp(iris[1], 0.0f, 63.0f);
    return cv::Point2f(cx, cy);
  } catch (const Ort::Exception &e) {
    std::cerr << "[Iris] Inferencia falló: " << e.what() << std::endl;
    return kFallback;
  }
}

void VisionPipeline::resetTemporalState() {
  landmarkFilter_.reset();
  hasLastLandmarkTimestamp_ = false;
}

bool VisionPipeline::estimateHeadPose(
    const OneEuroLandmarkFilter::LandmarkArray &stabilizedLandmarks,
    int frameWidth, int frameHeight, EulerAnglesDeg &headPoseDeg) const {
  // Standard 6-point set for 68-landmark models.
  //   30=nose tip, 8=chin, 36=left eye outer, 45=right eye outer,
  //   48=mouth left, 54=mouth right
  const std::array<int, 6> landmarkIndices = {30, 8, 36, 45, 48, 54};
  const std::array<cv::Point3d, 6> modelPoints = {
      cv::Point3d(  0.0, -330.0,  -65.0),   // 30: nose tip
      cv::Point3d(  0.0,  330.0,  -15.0),   // 8:  chin
      cv::Point3d(-225.0, -170.0, -135.0),  // 36: left eye outer corner
      cv::Point3d( 225.0, -170.0, -135.0),  // 45: right eye outer corner
      cv::Point3d(-150.0,  150.0, -125.0),  // 48: mouth left corner
      cv::Point3d( 150.0,  150.0, -125.0)}; // 54: mouth right corner

  std::vector<cv::Point2d> imagePoints;
  imagePoints.reserve(landmarkIndices.size());
  for (int index : landmarkIndices) {
    imagePoints.emplace_back(stabilizedLandmarks[index * 2],
                             stabilizedLandmarks[index * 2 + 1]);
  }

  const double focalLength = static_cast<double>(std::max(frameWidth, frameHeight));
  const cv::Point2d principalPoint(frameWidth / 2.0, frameHeight / 2.0);
  const cv::Mat cameraMatrix =
      (cv::Mat_<double>(3, 3) << focalLength, 0.0, principalPoint.x, 0.0,
       focalLength, principalPoint.y, 0.0, 0.0, 1.0);
  const cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

  cv::Mat rvec;
  cv::Mat tvec;
  if (!cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec,
                    tvec, false, cv::SOLVEPNP_ITERATIVE)) {
    return false;
  }

  cv::Mat rotationMatrix;
  cv::Rodrigues(rvec, rotationMatrix);

  const double sy = std::sqrt(rotationMatrix.at<double>(0, 0) *
                                  rotationMatrix.at<double>(0, 0) +
                              rotationMatrix.at<double>(1, 0) *
                                  rotationMatrix.at<double>(1, 0));
  const bool singular = sy < 1e-6;

  double pitch = 0.0;
  double yaw = 0.0;
  double roll = 0.0;
  if (!singular) {
    pitch = std::atan2(rotationMatrix.at<double>(2, 1),
                       rotationMatrix.at<double>(2, 2));
    yaw = std::atan2(-rotationMatrix.at<double>(2, 0), sy);
    roll = std::atan2(rotationMatrix.at<double>(1, 0),
                      rotationMatrix.at<double>(0, 0));
  } else {
    pitch = std::atan2(-rotationMatrix.at<double>(1, 2),
                       rotationMatrix.at<double>(1, 1));
    yaw = std::atan2(-rotationMatrix.at<double>(2, 0), sy);
  }

  constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
  headPoseDeg.pitch = pitch * kRadToDeg;
  headPoseDeg.yaw = yaw * kRadToDeg;
  headPoseDeg.roll = roll * kRadToDeg;
  return true;
}

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

  if (!effectEnabled || degradationLevel >= 3 || frame.empty()) {
    resetTemporalState();
    return;
  }

  // --- Diagnostic timing ---
  const auto t0 = std::chrono::high_resolution_clock::now();

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
    const auto t1 = std::chrono::high_resolution_clock::now();

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

    if (best_score < 0.7f || best_idx == -1) {
      resetTemporalState();
      return;
    }

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

    if (face_w < 10 || face_h < 10) {
      resetTemporalState();
      return;
    }

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
    const auto t2 = std::chrono::high_resolution_clock::now();

    const float *landmarks = pfldOutputTensors[0].GetTensorData<float>();
    OneEuroLandmarkFilter::LandmarkArray absoluteLandmarks{};
    for (std::size_t i = 0; i < OneEuroLandmarkFilter::kLandmarkCount; ++i) {
      float lx = landmarks[i * 2];
      float ly = landmarks[i * 2 + 1];
      absoluteLandmarks[i * 2] = face_x1 + lx * face_w;
      absoluteLandmarks[i * 2 + 1] = face_y1 + ly * face_h;
    }

    const auto now = std::chrono::steady_clock::now();
    double dtSeconds = nominalDtSeconds_;
    if (hasLastLandmarkTimestamp_) {
      dtSeconds = std::chrono::duration<double>(now - lastLandmarkTimestamp_)
                      .count();
      if (dtSeconds > 0.5) {
        landmarkFilter_.reset();
        dtSeconds = nominalDtSeconds_;
      }
    }
    lastLandmarkTimestamp_ = now;
    hasLastLandmarkTimestamp_ = true;

    const auto stabilizedLandmarks =
        landmarkFilter_.filterAbsolute(absoluteLandmarks, dtSeconds);
    EulerAnglesDeg headPoseDeg{};
    const bool hasHeadPose =
        estimateHeadPose(stabilizedLandmarks, w_f, h_f, headPoseDeg);

    auto create64x64Roi = [&](int start_idx, int end_idx) -> cv::Rect {
      float min_x = 1e9f, max_x = -1e9f;
      float min_y = 1e9f, max_y = -1e9f;

      for (int i = start_idx; i <= end_idx; ++i) {
        float abs_x = stabilizedLandmarks[i * 2];
        float abs_y = stabilizedLandmarks[i * 2 + 1];

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

    currentLeftRoi  = create64x64Roi(36, 41);  // left eye:  6 pts (dlib-68 convention)
    currentRightRoi = create64x64Roi(42, 47);  // right eye: 6 pts

    if (currentLeftRoi.area() > 0 && currentRightRoi.area() > 0) {
      validEyes = true;

      if (hasHeadPose) {
        float lastHRatio = 0.5f;
        float lastVRatio = 0.5f;
        double irisInfMs = 0.0;

        static cv::Point2f smoothedLeftGaze(0.5f, 0.5f);
        static cv::Point2f smoothedRightGaze(0.5f, 0.5f);
        constexpr float kSmoothingAlpha = 0.4f;

        auto applyEye = [&](const cv::Rect &roi, int startIdx, int endIdx) {
          cv::Mat eyeCrop = frame(roi).clone();

          // Extract landmarks in ROI-local coordinates.
          std::vector<cv::Point> localPts;
          localPts.reserve(endIdx - startIdx + 1);
          for (int i = startIdx; i <= endIdx; ++i) {
            int lx = std::clamp(static_cast<int>(std::round(
                stabilizedLandmarks[i * 2] - roi.x)), 0, 63);
            int ly = std::clamp(static_cast<int>(std::round(
                stabilizedLandmarks[i * 2 + 1] - roi.y)), 0, 63);
            localPts.push_back({lx, ly});
          }

          // Eye corners in absolute frame coordinates.
          // 68-landmark: left eye outer=36, inner=39; right eye outer=42, inner=45.
          int outerIdx = startIdx;      // 36 or 42: outer corner
          int innerIdx = startIdx + 3;  // 39 or 45: inner corner

          cv::Point2f outerCorner(stabilizedLandmarks[outerIdx * 2],
                                  stabilizedLandmarks[outerIdx * 2 + 1]);
          cv::Point2f innerCorner(stabilizedLandmarks[innerIdx * 2],
                                  stabilizedLandmarks[innerIdx * 2 + 1]);

          // Vertical centre of the eye (mean of upper and lower eyelid pairs).
          float upperY = (stabilizedLandmarks[(startIdx + 1) * 2 + 1] +
                          stabilizedLandmarks[(startIdx + 2) * 2 + 1]) * 0.5f;
          float lowerY = (stabilizedLandmarks[(startIdx + 4) * 2 + 1] +
                          stabilizedLandmarks[(startIdx + 5) * 2 + 1]) * 0.5f;

          // Detect iris centre using MediaPipe model; convert to absolute frame coords.
          const auto tIris0 = std::chrono::high_resolution_clock::now();
          cv::Point2f pupilLocal = detectIrisPupil(eyeCrop);
          const auto tIris1 = std::chrono::high_resolution_clock::now();
          irisInfMs += std::chrono::duration<double, std::milli>(tIris1 - tIris0).count();
          cv::Point2f pupilAbs(pupilLocal.x + roi.x, pupilLocal.y + roi.y);

          // Horizontal gaze ratio: 0.0 = outer corner, 1.0 = inner corner.
          float eyeWidth = cv::norm(innerCorner - outerCorner);
          float hRatio = 0.5f;
          if (eyeWidth > 5.0f) {
            cv::Point2f eyeVec   = innerCorner - outerCorner;
            cv::Point2f pupilVec = pupilAbs    - outerCorner;
            hRatio = (pupilVec.x * eyeVec.x + pupilVec.y * eyeVec.y) /
                     (eyeVec.x  * eyeVec.x  + eyeVec.y  * eyeVec.y);
            hRatio = std::clamp(hRatio, 0.0f, 1.0f);
          }

          // Vertical gaze ratio: 0.0 = upper lid, 1.0 = lower lid.
          float eyeHeight = lowerY - upperY;
          float vRatio = 0.5f;
          if (eyeHeight > 3.0f) {
            vRatio = (pupilAbs.y - upperY) / eyeHeight;
            vRatio = std::clamp(vRatio, 0.0f, 1.0f);
          }

          // Exponential smoothing of gaze ratios.
          cv::Point2f &smoothedGaze =
              (startIdx == 36) ? smoothedLeftGaze : smoothedRightGaze;
          smoothedGaze.x = kSmoothingAlpha * hRatio +
                           (1.0f - kSmoothingAlpha) * smoothedGaze.x;
          smoothedGaze.y = kSmoothingAlpha * vRatio +
                           (1.0f - kSmoothingAlpha) * smoothedGaze.y;

          // Deviation from 0.5 → pixel displacement in the 64×64 crop.
          constexpr float kHorizontalStrength = 40.0f;
          constexpr float kVerticalStrength   = 35.0f;

          // Punto neutral calibrado empíricamente
          // (valores medidos cuando el usuario mira directamente a la cámara)
          constexpr float kNeutralH = 0.54f;
          constexpr float kNeutralV = 0.35f;

          float dx = (kNeutralH - smoothedGaze.x) * kHorizontalStrength;
          float dy = (kNeutralV - smoothedGaze.y) * kVerticalStrength;
          dx = std::clamp(dx, -14.0f, 14.0f);
          dy = std::clamp(dy, -14.0f, 14.0f);

          cv::Mat corrected =
              geometryEngine_.applyDisplacement(eyeCrop, localPts, dx, dy);
          if (!corrected.empty())
            corrected.copyTo(frame(roi));

          lastHRatio = smoothedGaze.x;
          lastVRatio = smoothedGaze.y;
        };

        applyEye(currentLeftRoi,  36, 41);
        applyEye(currentRightRoi, 42, 47);
        const auto t3 = std::chrono::high_resolution_clock::now();

        static int diagCounter = 0;
        if (++diagCounter % 30 == 0) {
          using ms = std::chrono::duration<double, std::milli>;
          std::cout << "[DIAG] UltraFace: " << ms(t1 - t0).count()
                    << "ms | PFLD: "          << ms(t2 - t1).count()
                    << "ms | Iris: "          << irisInfMs
                    << "ms | GeometryEngine: " << ms(t3 - t2).count() - irisInfMs
                    << "ms | Total: "         << ms(t3 - t0).count()
                    << "ms | Gaze: h="        << lastHRatio
                    << " v=" << lastVRatio << std::endl;
        }
      }
    }



  } catch (const Ort::Exception &e) {
    std::cerr << "[Pipeline] Hardware ONNX abortado geométricamente: "
              << e.what() << std::endl;
    validEyes = false;
    resetTemporalState();
  } catch (const cv::Exception &e) {
    std::cerr << "[Pipeline] OpenCV abortado geométricamente: " << e.what()
              << std::endl;
    validEyes = false;
    resetTemporalState();
  }
}
