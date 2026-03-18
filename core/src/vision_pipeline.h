#pragma once
#include <array>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class OneEuroFilter {
  float min_cutoff, beta, d_cutoff;
  float x_prev, dx_prev;
  bool initialized;
  float alpha(float cutoff, float dt) {
    float tau = 1.0f / (2.0f * 3.14159265f * cutoff);
    return 1.0f / (1.0f + tau / dt);
  }

public:
  OneEuroFilter(float mc = 1.0f, float b = 0.07f)
      : min_cutoff(mc), beta(b), d_cutoff(1.0f), x_prev(0), dx_prev(0),
        initialized(false) {}
  float filter(float x, float dt) {
    if (!initialized) {
      x_prev = x;
      initialized = true;
      return x;
    }
    float dx = (x - x_prev) / dt;
    float a_d = alpha(d_cutoff, dt);
    dx_prev += a_d * (dx - dx_prev);
    float cutoff = min_cutoff + beta * std::abs(dx_prev);
    float a = alpha(cutoff, dt);
    x_prev += a * (x - x_prev);
    return x_prev;
  }
};

class VisionPipeline {
public:
  VisionPipeline(const std::string &faceModelPath, const std::string &modelPath,
                 double fps = 30.0);
  void process(cv::Mat &frame, bool effectEnabled, int degradationLevel,
               float dt);

private:
  void preprocessFrame(const cv::Mat &frame, cv::Mat &resized,
                       cv::Mat &blobBuffer, float *buffer);
  std::vector<std::array<float, 4>> generateUltraFacePriors(int imgW, int imgH);

  // Architectural Constants
  static constexpr int MODEL_SIZE = 112;
  static constexpr int TENSOR_ELEMENTS = 1 * 3 * MODEL_SIZE * MODEL_SIZE;
  static constexpr int UF_W = 320;
  static constexpr int UF_H = 240;
  static constexpr int UF_ELEMENTS = 1 * 3 * UF_H * UF_W;

  // Ort Configurations
  Ort::Env env;
  Ort::SessionOptions sessionOpts;
  Ort::RunOptions runOpts;

  // Face Model (UltraFace)
  std::unique_ptr<Ort::Session> faceSession;
  bool faceDetectorOk{false};
  std::vector<float> ufTensorData;
  std::vector<std::array<float, 4>> ufPriors;
  Ort::Value ufInputTensor{nullptr};
  std::string ufInputNameStr;
  const char *ufInputName{nullptr};
  std::vector<std::string> ufOutNamesStr;
  std::vector<const char *> ufOutputNames;

  // Landmarks Model (PFLD)
  std::unique_ptr<Ort::Session> session;
  bool modelLoaded{false};
  std::vector<float> tensorData;
  Ort::Value inputTensor{nullptr};
  std::string cachedInputNameStr;
  const char *cachedInputName{nullptr};
  std::string cachedOutputNameStr;
  const char *cachedOutputName{nullptr};

  // OpenCV Buffers
  cv::Mat resized;
  cv::Mat ufResized;
  cv::Mat blobBuffer;

  // State Flags
  bool isBlinking{false};
  bool headOutOfBounds{false};
  float warpMultiplier{0.0f};
  int frameCounter{0};
  float temporalStep{0.0f};

  // State Machine for Thermal Degradation (Level 2 Momentum)
  std::vector<float> previousLandmarks;
  std::vector<float> currentLandmarks;
  bool isSkippedFrame{false};

  // State Machine for Thermal Degradation (Level 1 UltraFace Cache)
  int ufSkipCounter{0};
  cv::Rect lastRoi;
  bool hasValidLastRoi{false};

  // Edge-triggered logging state
  bool prevFaceFound{false};
  bool prevWarpFullyActive{false};
  bool prevBlinking{false};

  // EMA temporal filter state (per-eye shift persistence)
  float last_shift_lx{0.0f};
  float last_shift_ly{0.0f};
  float last_shift_rx{0.0f};
  float last_shift_ry{0.0f};

  // Face detection hysteresis
  int face_loss_buffer{0};

  // OneEuroFilter for head pose angles
  OneEuroFilter pitchFilter{1.0f, 0.07f};
  OneEuroFilter yawFilter{1.0f, 0.07f};
  OneEuroFilter rollFilter{1.0f, 0.07f};
  float filteredPitch{0.0f};
  float filteredYaw{0.0f};
  float filteredRoll{0.0f};

  // OneEuroFilter for ocular bounding boxes (Absolute Space)
  OneEuroFilter leftEyeXFilter{1.0f, 0.07f};
  OneEuroFilter leftEyeYFilter{1.0f, 0.07f};
  OneEuroFilter leftEyeWFilter{1.0f, 0.07f};
  OneEuroFilter leftEyeHFilter{1.0f, 0.07f};

  OneEuroFilter rightEyeXFilter{1.0f, 0.07f};
  OneEuroFilter rightEyeYFilter{1.0f, 0.07f};
  OneEuroFilter rightEyeWFilter{1.0f, 0.07f};
  OneEuroFilter rightEyeHFilter{1.0f, 0.07f};

  // Asymmetric hysteresis state machine
  bool effectActive{true};
  int reentryCounter{0};
};
