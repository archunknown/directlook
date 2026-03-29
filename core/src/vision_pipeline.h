// vision_pipeline.h
#pragma once

#include "temporal_filter.hpp"
#include <chrono>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Migrated from geometry_engine.hpp — still needed by estimateHeadPose.
struct EulerAnglesDeg {
  double pitch{0.0};
  double yaw{0.0};
  double roll{0.0};
};

#include "geometry_engine.hpp"

class VisionPipeline {
public:
  VisionPipeline(const std::string &faceModelPath, const std::string &modelPath,
                 const std::string &irisModelPath, double fps = 30.0);
  ~VisionPipeline();

  void process(cv::Mat &frame, bool effectEnabled, int degradationLevel);

  bool hasValidEyes() const;
  cv::Rect getLeftEyeRoi() const;
  cv::Rect getRightEyeRoi() const;

private:
  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> faceSession;
  std::unique_ptr<Ort::Session> session;

  std::vector<float> faceInputTensorValues;
  std::string faceInputNameStr;
  const char *faceInputName{nullptr};

  std::vector<std::string> faceOutputNamesStr;
  std::vector<const char *> faceOutputNames;

  std::vector<float> pfldInputTensorValues;
  std::string pfldInputNameStr;
  const char *pfldInputName{nullptr};

  std::string pfldOutputNameStr;
  const char *pfldOutputName{nullptr};

  // Iris landmark session (MediaPipe iris_landmark.onnx — [1,3,64,64] → [1,15])
  std::unique_ptr<Ort::Session> irisSession;
  std::vector<float>            irisInputTensorValues; // 3 × 64 × 64
  std::string irisInputNameStr;
  const char *irisInputName{nullptr};
  std::string irisOutputNameStr;  // "output_iris" → float[1,15]
  const char *irisOutputName{nullptr};

  cv::Point2f detectIrisPupil(const cv::Mat &eyeCrop64) const;

  std::vector<std::vector<float>> priors;
  void generatePriors(int img_w, int img_h);
  void resetTemporalState();
  bool estimateHeadPose(
      const OneEuroLandmarkFilter::LandmarkArray &stabilizedLandmarks,
      int frameWidth, int frameHeight, EulerAnglesDeg &headPoseDeg) const;

  cv::Rect currentLeftRoi;
  cv::Rect currentRightRoi;
  bool validEyes{false};
  GeometryEngine geometryEngine_;
  OneEuroLandmarkFilter landmarkFilter_;
  double nominalDtSeconds_{1.0 / 30.0};
  std::chrono::steady_clock::time_point lastLandmarkTimestamp_{};
  bool hasLastLandmarkTimestamp_{false};
};
