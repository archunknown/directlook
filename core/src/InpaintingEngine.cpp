#include "InpaintingEngine.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

InpaintingEngine::InpaintingEngine(const std::string &modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "DirectLookInpainting"),
      inputTensorValues(TENSOR_ELEMENTS, 0.0f) {

  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  try {
#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    session =
        std::make_unique<Ort::Session>(env, wpath.c_str(), sessionOptions);
#else
    session =
        std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions alloc;

    auto inNameAllocated = session->GetInputNameAllocated(0, alloc);
    inputNameStr = inNameAllocated.get();
    inputName = inputNameStr.c_str();

    auto outNameAllocated = session->GetOutputNameAllocated(0, alloc);
    outputNameStr = outNameAllocated.get();
    outputName = outputNameStr.c_str();

    std::cout << "[ONNX] Inpainting Engine inicializado: " << modelPath
              << std::endl;
  } catch (const Ort::Exception &e) {
    std::cerr << "[ONNX] Falla crítica cargando inpainting_fp32.onnx: "
              << e.what() << std::endl;
    throw;
  }
}

cv::Mat InpaintingEngine::processEye(cv::Mat &eyeCrop) {
  if (eyeCrop.empty() || eyeCrop.cols != INPAINT_SIZE ||
      eyeCrop.rows != INPAINT_SIZE) {
    return cv::Mat();
  }

  cv::Mat rgb_eye;
  cv::cvtColor(eyeCrop, rgb_eye, cv::COLOR_BGR2RGB);

  cv::Mat rgb_float;
  rgb_eye.convertTo(rgb_float, CV_32FC3, 1.0f / 255.0f);

  const float *src = reinterpret_cast<const float *>(rgb_float.data);
  float *dst = inputTensorValues.data();
  const int hw = INPAINT_SIZE * INPAINT_SIZE;

  for (int i = 0; i < hw; ++i) {
    dst[i] = src[i * 3 + 0];          // R
    dst[i + hw] = src[i * 3 + 1];     // G
    dst[i + hw * 2] = src[i * 3 + 2]; // B
  }

  const char *inputNames[] = {inputName};
  const char *outputNames[] = {outputName};
  const std::array<int64_t, 4> inputShape = {1, CHANNELS, INPAINT_SIZE,
                                             INPAINT_SIZE};
  const auto memInfo =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  try {
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    auto outputTensors =
        session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                     outputNames, 1);
    const float *out_data = outputTensors[0].GetTensorData<float>();

    cv::Mat out_rgb_float(INPAINT_SIZE, INPAINT_SIZE, CV_32FC3);
    float *outDst = reinterpret_cast<float *>(out_rgb_float.data);

    for (int i = 0; i < hw; ++i) {
      outDst[i * 3 + 0] = out_data[i];          // R
      outDst[i * 3 + 1] = out_data[i + hw];     // G
      outDst[i * 3 + 2] = out_data[i + hw * 2]; // B
    }

    cv::Mat out_bgr_float;
    cv::Mat out_bgr_8u;
    cv::cvtColor(out_rgb_float, out_bgr_float, cv::COLOR_RGB2BGR);
    out_bgr_float.convertTo(out_bgr_8u, CV_8UC3, 255.0f);

    return out_bgr_8u;
  } catch (const Ort::Exception &e) {
    std::cerr << "[InpaintingEngine] Error durante inferencia: " << e.what()
              << std::endl;
    return cv::Mat();
  }
}
