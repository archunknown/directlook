#pragma once
#include <array>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class VisionPipeline {
public:
  VisionPipeline(const std::string &faceModelPath,
                 const std::string &modelPath);
  void process(cv::Mat &frame, bool effectEnabled);

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
};
