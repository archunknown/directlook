#include "vision_pipeline.h"
#include <cmath>
#include <iostream>

VisionPipeline::VisionPipeline(const std::string &faceModelPath,
                               const std::string &modelPath, double fps)
    : env(ORT_LOGGING_LEVEL_WARNING, "DirectLookDaemon"),
      ufTensorData(UF_ELEMENTS), tensorData(TENSOR_ELEMENTS),
      previousLandmarks(98 * 2, 0.0f), currentLandmarks(98 * 2, 0.0f) {

  temporalStep =
      static_cast<float>(1.0 / (fps * 0.1)); // Transición en 0.1 seg (100ms)

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

    size_t outCount = faceSession->GetOutputCount();
    ufOutNamesStr.reserve(outCount);
    ufOutputNames.reserve(outCount);
    for (size_t i = 0; i < outCount; ++i) {
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

void VisionPipeline::process(cv::Mat &frame, bool effectEnabled,
                             int degradationLevel) {
  cv::Rect roi;
  bool faceFound = false;

  if (effectEnabled && faceDetectorOk && faceSession) {
    bool runFaceDetection = true;
    if (degradationLevel >= 3) {
      runFaceDetection = false;
    } else if (degradationLevel >= 1) {
      if (ufSkipCounter++ % 4 != 0 && hasValidLastRoi) {
        runFaceDetection = false;
      }
    } else {
      ufSkipCounter = 0;
    }

    if (runFaceDetection) {
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

            // Cache para el Nivel 1 de QoS térmico
            lastRoi = roi;
            hasValidLastRoi = true;

            faceFound = true;
          }
        }
      } catch (const Ort::Exception &e) {
        std::cerr << "[Inferencia UltraFace Error] " << e.what() << std::endl;
        return;
      }
    } else if (hasValidLastRoi) {
      // Nivel 1: Evitamos el cálculo UltraFace y reusamos la caja anterior
      roi = lastRoi;
      faceFound = true;
    }
  }

  if (faceFound) {
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    cv::Mat faceCrop = frame(roi);

    int currentSize = MODEL_SIZE; // Constant 112x112 for ONNX static shape
    cv::resize(faceCrop, resized, cv::Size(currentSize, currentSize));
    cv::dnn::blobFromImage(resized, blobBuffer, 1.0 / 255.0, cv::Size(),
                           cv::Scalar(), true, false);
    std::memcpy(tensorData.data(), blobBuffer.ptr<float>(),
                blobBuffer.total() * sizeof(float));

    if (modelLoaded && session) {
      try {
        if (degradationLevel >= 2 && degradationLevel < 3) {
          isSkippedFrame = !isSkippedFrame;
        } else if (degradationLevel >= 3) {
          isSkippedFrame = true;
        } else {
          isSkippedFrame = false;
        }

        const float *landmarks = currentLandmarks.data();

        if (isSkippedFrame) {
          if (degradationLevel >= 3) {
            currentLandmarks = previousLandmarks;
          } else {
            // Nivel 2: Extrapolación lineal / Momentum pre-proyectivo
            const float MAX_DISPLACEMENT = 0.05f; // Clamping Físico Normalizado
          for (int i = 0; i < 98; ++i) {
            float deltaX = currentLandmarks[i * 2] - previousLandmarks[i * 2];
            float deltaY = currentLandmarks[i * 2 + 1] - previousLandmarks[i * 2 + 1];
            
            float mag = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            if (mag > MAX_DISPLACEMENT) {
              // Movimiento irracional detectado. Anular extrapolación (Mantiene current = anterior salto).
            } else {
              currentLandmarks[i * 2] = currentLandmarks[i * 2] + deltaX;
            }
          }
        }
        } else if (degradationLevel < 3) {
          previousLandmarks = currentLandmarks;

          std::array<int64_t, 4> dynShape = {1, 3, currentSize, currentSize};
          auto memInfo =
              Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
          auto dynInput = Ort::Value::CreateTensor<float>(
              memInfo, tensorData.data(),
              dynShape[0] * dynShape[1] * dynShape[2] * dynShape[3],
              dynShape.data(), dynShape.size());

          auto results = session->Run(runOpts, &cachedInputName, &dynInput, 1,
                                      &cachedOutputName, 1);

          const float *outData = results[0].GetTensorData<float>();
          for (int i = 0; i < 98 * 2; ++i) {
            currentLandmarks[i] = outData[i];
          }
        }

        // Cálculo de EAR (Eye Aspect Ratio) para detección de parpadeo
        auto dist = [&](int p1, int p2) {
          float dx = landmarks[p1 * 2] - landmarks[p2 * 2];
          float dy = landmarks[p1 * 2 + 1] - landmarks[p2 * 2 + 1];
          return std::sqrt(dx * dx + dy * dy);
        };

        float leftEAR = (dist(61, 67) + dist(62, 66) + dist(63, 65)) /
                        (3.0f * dist(60, 64));
        float rightEAR = (dist(69, 75) + dist(70, 74) + dist(71, 73)) /
                         (3.0f * dist(68, 72));
        float avgEAR = (leftEAR + rightEAR) / 2.0f;

        const float BLINK_THRESHOLD = 0.2f;
        isBlinking = (avgEAR < BLINK_THRESHOLD);

        frameCounter++;

        // Estimación de Pose Cefálica (Head Pose Estimation) - Frame Skipping
        // c/4 frames, solo calculable si la matriz landmark es real y no extrapolada
        if (!isSkippedFrame && frameCounter % 4 == 0) {
          std::vector<cv::Point2f> image_points;
          auto get_pt = [&](int idx) {
            return cv::Point2f(roi.x + landmarks[idx * 2] * roi.width,
                               roi.y + landmarks[idx * 2 + 1] * roi.height);
          };
          // 54: Nariz, 16: Barbilla, 60: Esquina exterior ojo izq, 72: Esquina
          // exterior ojo der, 76: Comisura boca izq, 82: Comisura boca der
          image_points.push_back(get_pt(54));
          image_points.push_back(get_pt(16));
          image_points.push_back(get_pt(60));
          image_points.push_back(get_pt(72));
          image_points.push_back(get_pt(76));
          image_points.push_back(get_pt(82));

          std::vector<cv::Point3f> model_points;
          model_points.push_back(cv::Point3f(0.0f, 0.0f, 0.0f)); // Nariz
          model_points.push_back(
              cv::Point3f(0.0f, -330.0f, -65.0f)); // Barbilla
          model_points.push_back(
              cv::Point3f(-225.0f, 170.0f, -135.0f)); // Ojo izq ext
          model_points.push_back(
              cv::Point3f(225.0f, 170.0f, -135.0f)); // Ojo der ext
          model_points.push_back(
              cv::Point3f(-150.0f, -150.0f, -125.0f)); // Boca izq
          model_points.push_back(
              cv::Point3f(150.0f, -150.0f, -125.0f)); // Boca der

          double focal_length = frame.cols;
          cv::Point2d center = cv::Point2d(frame.cols / 2.0, frame.rows / 2.0);
          cv::Mat camera_matrix =
              (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0,
               focal_length, center.y, 0, 0, 1);
          cv::Mat dist_coeffs =
              cv::Mat::zeros(4, 1, cv::DataType<double>::type);

          cv::Mat rotation_vector, translation_vector;
          cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                       rotation_vector, translation_vector);

          cv::Mat rotation_matrix;
          cv::Rodrigues(rotation_vector, rotation_matrix);

          cv::Mat mtxR, mtxQ, Qx, Qy, Qz;
          cv::Vec3d eulerAngles =
              cv::RQDecomp3x3(rotation_matrix, mtxR, mtxQ, Qx, Qy, Qz);

          // Pitch = eulerAngles[0], Yaw = eulerAngles[1], Roll = eulerAngles[2]
          headOutOfBounds = (std::abs(eulerAngles[0]) > 15.0 ||
                             std::abs(eulerAngles[1]) > 15.0);
        }
        // Nivel 3: Bypass de capa visual y Apagado Inmediato Estético
        bool shouldApplyEffect = effectEnabled && !isBlinking &&
                                 !headOutOfBounds && (degradationLevel < 3);
        if (degradationLevel >= 3) {
          warpMultiplier = 0.0f;
        } else if (shouldApplyEffect) {
          warpMultiplier =
              std::min(1.0f, warpMultiplier +
                                 temporalStep); // Sube 0.0 a 1.0 iterativamente
        } else {
          warpMultiplier =
              std::max(0.0f, warpMultiplier -
                                 temporalStep); // Baja 1.0 a 0.0 iterativamente
        }

        if (warpMultiplier > 0.0f) {
          // Extracción de sub-fragmentos oculares confinados y marginados
          auto get_eye_roi = [&](int start_idx, int end_idx) {
            int min_x = frame.cols, max_x = 0;
            int min_y = frame.rows, max_y = 0;
            for (int i = start_idx; i <= end_idx; ++i) {
              int px = roi.x + static_cast<int>(landmarks[i * 2] * roi.width);
              int py =
                  roi.y + static_cast<int>(landmarks[i * 2 + 1] * roi.height);
              if (px < min_x)
                min_x = px;
              if (px > max_x)
                max_x = px;
              if (py < min_y)
                min_y = py;
              if (py > max_y)
                max_y = py;
            }
            int w = max_x - min_x;
            int h = max_y - min_y;
            int pad_x = static_cast<int>(w * 0.2f);
            int pad_y = static_cast<int>(h * 0.2f);
            min_x = std::max(0, min_x - pad_x);
            min_y = std::max(0, min_y - pad_y);
            max_x = std::min(frame.cols - 1, max_x + pad_x);
            max_y = std::min(frame.rows - 1, max_y + pad_y);
            return cv::Rect(min_x, min_y, std::max(1, max_x - min_x),
                            std::max(1, max_y - min_y));
          };

          cv::Rect leftEyeRoi = get_eye_roi(60, 67);
          cv::Rect rightEyeRoi = get_eye_roi(68, 75);

          cv::Mat left_map_x(leftEyeRoi.size(), CV_32FC1);
          cv::Mat left_map_y(leftEyeRoi.size(), CV_32FC1);
          cv::Mat right_map_x(rightEyeRoi.size(), CV_32FC1);
          cv::Mat right_map_y(rightEyeRoi.size(), CV_32FC1);

          auto apply_spherical_warp = [&](cv::Rect eyeRoi, cv::Mat &map_x,
                                          cv::Mat &map_y) {
            float cx = eyeRoi.width / 2.0f;
            float cy = eyeRoi.height / 2.0f;
            float radius = std::min(cx, cy);
            float max_shift = radius * 0.5f;

            for (int y = 0; y < eyeRoi.height; ++y) {
              float *ptr_x = map_x.ptr<float>(y);
              float *ptr_y = map_y.ptr<float>(y);
              for (int x = 0; x < eyeRoi.width; ++x) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx * dx + dy * dy);

                float shift = 0.0f;
                // Deformación Gaussiana/Parabólica confinada al radio espacial
                if (dist < radius) {
                  float curve = 1.0f - (dist * dist) / (radius * radius);
                  shift = max_shift * curve;
                }

                ptr_x[x] = static_cast<float>(x);
                ptr_y[y] =
                    static_cast<float>(y) -
                    shift; // Resta explícita para el desplazamiento vertical
              }
            }

            cv::Mat warped_eye;
            cv::Mat eye_crop = frame(eyeRoi);
            cv::remap(eye_crop, warped_eye, map_x, map_y, cv::INTER_LINEAR);

            // Alpha Blending con base en el multiplicador estricto temporal
            cv::addWeighted(warped_eye, warpMultiplier, eye_crop,
                            1.0f - warpMultiplier, 0.0, eye_crop);
          };

          // Ejecución Secuencial Independiente
          apply_spherical_warp(leftEyeRoi, left_map_x, left_map_y);
          apply_spherical_warp(rightEyeRoi, right_map_x, right_map_y);
        }

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
