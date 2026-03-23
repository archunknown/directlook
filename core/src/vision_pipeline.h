// vision_pipeline.h
#pragma once

#include "geometry_engine.hpp"
#include "temporal_filter.hpp"
#include <chrono>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VisionPipeline {
public:
  VisionPipeline(const std::string &faceModelPath, const std::string &modelPath,
                 double fps = 30.0);
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
