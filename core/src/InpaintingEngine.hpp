#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

class InpaintingEngine {
public:
    explicit InpaintingEngine(const std::string& modelPath);
    ~InpaintingEngine() = default;

    cv::Mat processEye(cv::Mat& eyeCrop);

private:
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;

    std::vector<float> inputTensorValues;

    std::string inputNameStr;
    const char* inputName{nullptr};
    std::string outputNameStr;
    const char* outputName{nullptr};

    static constexpr int INPAINT_SIZE = 64;
    static constexpr int CHANNELS = 3;
    static constexpr int TENSOR_ELEMENTS = CHANNELS * INPAINT_SIZE * INPAINT_SIZE;
};
