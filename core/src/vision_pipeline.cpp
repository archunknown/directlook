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
                             int degradationLevel, float dt) {
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

  // Histéresis de detección: buffer de 10 frames antes de declarar pérdida
  if (faceFound) {
    face_loss_buffer = 0;
  } else if (hasValidLastRoi && face_loss_buffer < 10) {
    face_loss_buffer++;
    roi = lastRoi;
    faceFound = true;
  }

  if (faceFound) {
    if (!prevFaceFound) {
      std::cout << "[DAEMON] Rostro enganchado\n";
    }

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
              float deltaY =
                  currentLandmarks[i * 2 + 1] - previousLandmarks[i * 2 + 1];

              float mag = std::sqrt(deltaX * deltaX + deltaY * deltaY);
              if (mag > MAX_DISPLACEMENT) {
                // Movimiento irracional detectado. Anular extrapolación
                // (Mantiene current = anterior salto).
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
        // c/4 frames, solo calculable si la matriz landmark es real y no
        // extrapolada
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
              cv::Point3f(0.0f, 330.0f, -65.0f)); // Barbilla (Y+)
          model_points.push_back(
              cv::Point3f(-225.0f, -170.0f, -135.0f)); // Ojo izq (Y-)
          model_points.push_back(
              cv::Point3f(225.0f, -170.0f, -135.0f)); // Ojo der (Y-)
          model_points.push_back(
              cv::Point3f(-150.0f, 150.0f, -125.0f)); // Boca izq (Y+)
          model_points.push_back(
              cv::Point3f(150.0f, 150.0f, -125.0f)); // Boca der (Y+)

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

          // OneEuroFilter sobre Pitch, Yaw, Roll con dt real
          filteredPitch =
              pitchFilter.filter(static_cast<float>(eulerAngles[0]), dt);
          filteredYaw =
              yawFilter.filter(static_cast<float>(eulerAngles[1]), dt);
          filteredRoll =
              rollFilter.filter(static_cast<float>(eulerAngles[2]), dt);

          // Histéresis asimétrica: desactivar a >20°, reactivar si <10° por 8
          // frames
          if (std::abs(filteredPitch) > 20.0f ||
              std::abs(filteredYaw) > 20.0f) {
            effectActive = false;
            reentryCounter = 0;
          } else if (!effectActive) {
            if (std::abs(filteredPitch) < 10.0f &&
                std::abs(filteredYaw) < 10.0f) {
              if (++reentryCounter >= 8) {
                effectActive = true;
              }
            } else {
              reentryCounter = 0;
            }
          }
          headOutOfBounds = !effectActive;
        }
        bool shouldApplyEffect = effectEnabled && !isBlinking && effectActive &&
                                 (degradationLevel < 3);
        if (degradationLevel >= 3) {
          warpMultiplier = 0.0f;
        } else if (shouldApplyEffect) {
          warpMultiplier = 1.0f; // Salto inmediato: autoridad total restaurada
        } else {
          warpMultiplier = std::max(0.0f, warpMultiplier - temporalStep);
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

          cv::Rect rawLeftEyeRoi = get_eye_roi(60, 67);
          cv::Rect rawRightEyeRoi = get_eye_roi(68, 75);

          auto apply_roi_filter = [&](cv::Rect raw, OneEuroFilter &fx,
                                      OneEuroFilter &fy, OneEuroFilter &fw,
                                      OneEuroFilter &fh) {
            int x = static_cast<int>(fx.filter(static_cast<float>(raw.x), dt));
            int y = static_cast<int>(fy.filter(static_cast<float>(raw.y), dt));
            int w =
                static_cast<int>(fw.filter(static_cast<float>(raw.width), dt));
            int h =
                static_cast<int>(fh.filter(static_cast<float>(raw.height), dt));

            x = std::max(0, std::min(x, frame.cols - 1));
            y = std::max(0, std::min(y, frame.rows - 1));
            if (x + w > frame.cols)
              w = frame.cols - x;
            if (y + h > frame.rows)
              h = frame.rows - y;
            w = std::max(1, w);
            h = std::max(1, h);

            return cv::Rect(x, y, w, h);
          };

          cv::Rect leftEyeRoi =
              apply_roi_filter(rawLeftEyeRoi, leftEyeXFilter, leftEyeYFilter,
                               leftEyeWFilter, leftEyeHFilter);
          cv::Rect rightEyeRoi =
              apply_roi_filter(rawRightEyeRoi, rightEyeXFilter, rightEyeYFilter,
                               rightEyeWFilter, rightEyeHFilter);

          cv::Mat left_map_x(leftEyeRoi.size(), CV_32FC1);
          cv::Mat left_map_y(leftEyeRoi.size(), CV_32FC1);
          cv::Mat right_map_x(rightEyeRoi.size(), CV_32FC1);
          cv::Mat right_map_y(rightEyeRoi.size(), CV_32FC1);

          const bool FLIP_CORRECTION =
              true; // Actívala o desactívala según el log

          auto apply_spherical_warp = [&](cv::Rect eyeRoi, cv::Mat &map_x,
                                          cv::Mat &map_y, float raw_shift_x,
                                          float raw_shift_y,
                                          const std::string &sideName) {
            float cx = eyeRoi.width / 2.0f;
            float cy = eyeRoi.height / 2.0f;
            float radius = std::max(cx * 1.2f, eyeRoi.height * 0.8f);

            float shift_x = raw_shift_x;
            float shift_y = raw_shift_y;

            // INVERSIÓN CONDICIONAL
            if (FLIP_CORRECTION) {
              shift_x = -shift_x;
              shift_y = -shift_y;
            }

            float mag = std::sqrt(shift_x * shift_x + shift_y * shift_y);
            float max_allowed = radius * 0.4f;

            if (mag > max_allowed && mag > 0.0f) {
              shift_x = (shift_x / mag) * max_allowed;
              shift_y = (shift_y / mag) * max_allowed;
            }

            // Sensibilidad Amplificada + Intensidad del Efecto (Factor
            // Dinámico)
            shift_x *= warpMultiplier * 1.5f;
            shift_y *= warpMultiplier * 1.5f;

            // Dinámica inercial mínima de prueba para romper inercia de la zona
            // muerta
            if (mag < radius * 0.04f) {
              shift_x = radius * 0.05f;
              shift_y = 0.0f;
            }

            // Auditoría Interna: Muestreo del píxel central de la ROI
            if (frameCounter % 30 == 0) {
              float dist_center = 0.0f; // Distancia en px céntrico es 0
              float factor_esferico =
                  1.0f - (dist_center * dist_center) / (radius * radius);
              float test_curve_x = shift_x * factor_esferico;
              float test_curve_y = shift_y * factor_esferico;

              std::cerr << "[" << sideName << " CENTER MAP] Shift vector = ("
                        << shift_x << ", " << shift_y << ") -> Curve output = ("
                        << test_curve_x << ", " << test_curve_y << ")"
                        << std::endl;
            }

            for (int y = 0; y < eyeRoi.height; ++y) {
              float *ptr_x = map_x.ptr<float>(y);
              float *ptr_y = map_y.ptr<float>(y);
              for (int x = 0; x < eyeRoi.width; ++x) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx * dx + dy * dy);

                float curve_x = 0.0f;
                float curve_y = 0.0f;
                if (dist < radius) {
                  float factor = 1.0f - (dist * dist) / (radius * radius);
                  curve_x = shift_x * factor;
                  curve_y = shift_y * factor;
                }

                ptr_x[x] = static_cast<float>(x) - curve_x;
                ptr_y[y] = static_cast<float>(y) - curve_y;
              }
            }

            // 1. Compensación de Roll (aislada)
#ifndef DIRECTLOOK_ROLL_MODE_ATTENUATION
            if (std::abs(filteredRoll) > 0.5f) {
              cv::Point2f center(map_x.cols / 2.0f, map_x.rows / 2.0f);
              cv::Mat rot = cv::getRotationMatrix2D(center, -filteredRoll, 1.0);

              cv::Mat temp_x, temp_y;
              cv::warpAffine(map_x, temp_x, rot, map_x.size(), cv::INTER_LINEAR,
                             cv::BORDER_REPLICATE);
              cv::warpAffine(map_y, temp_y, rot, map_y.size(), cv::INTER_LINEAR,
                             cv::BORDER_REPLICATE);

              map_x = temp_x;
              map_y = temp_y;
            }
#else
            // Plan B: Atenuación lineal del vector por Roll
            float roll_atten =
                std::max(0.0f, 1.0f - std::abs(filteredRoll) / 45.0f);
            shift_x *= roll_atten;
            shift_y *= roll_atten;
#endif

            // DEBUG: Vectores de corrección sobre el feed
            std::string text = "ShiftX: " + std::to_string(shift_x) +
                               " ShiftY: " + std::to_string(shift_y);
            cv::putText(frame, text, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0),
                        2);
            std::string text_rad = "Radio: " + std::to_string(radius) +
                                   " Mult: " + std::to_string(warpMultiplier);
            cv::putText(frame, text_rad, cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0),
                        2);

            // 2. Remapeo Esférico (Estrictamente out-of-place)
            cv::Mat temp_roi;
            cv::remap(frame(eyeRoi), temp_roi, map_x, map_y, cv::INTER_LINEAR,
                      cv::BORDER_REPLICATE);

            // 3. Aplicación visual con Alpha (warpMultiplier)
            if (warpMultiplier >= 0.999f) {
              temp_roi.copyTo(frame(eyeRoi));
            } else {
              cv::addWeighted(frame(eyeRoi), 1.0f - warpMultiplier, temp_roi,
                              warpMultiplier, 0.0, frame(eyeRoi));
            }
          };

          // --- Ojo Izquierdo: EMA + Deadzone ---
          float left_pupila_x_abs = roi.x + landmarks[97 * 2] * roi.width;
          float left_pupila_y_abs = roi.y + landmarks[97 * 2 + 1] * roi.height;
          float left_pupila_x_relativa =
              left_pupila_x_abs - static_cast<float>(leftEyeRoi.x);
          float left_pupila_y_relativa =
              left_pupila_y_abs - static_cast<float>(leftEyeRoi.y);
          float left_target_x = leftEyeRoi.width / 2.0f;
          float left_target_y = leftEyeRoi.height / 2.0f;
          float left_raw_shift_x = left_target_x - left_pupila_x_relativa;
          float left_raw_shift_y = left_target_y - left_pupila_y_relativa;

          float left_deadzone = 0.04f * static_cast<float>(leftEyeRoi.width);
          float left_delta_x = std::abs(left_raw_shift_x - last_shift_lx);
          float left_delta_y = std::abs(left_raw_shift_y - last_shift_ly);

          // Auditoría Externa Global (Cada 30 frames)
          if (frameCounter % 30 == 0) {
            std::cerr << "--- DIAGNOSTICO DE INVERSION [Frame " << frameCounter
                      << "] ---\n"
                      << "Angulos   : Pitch=" << filteredPitch
                      << ", Yaw=" << filteredYaw << "\n"
                      << "ROI Izq   : Width=" << leftEyeRoi.width
                      << ", Height=" << leftEyeRoi.height << "\n"
                      << "Target [L]: X=" << left_target_x
                      << " Y=" << left_target_y << "\n"
                      << "Actual [L]: X=" << left_pupila_x_relativa
                      << " Y=" << left_pupila_y_relativa << "\n"
                      << "Shift  [L]: DX=" << left_raw_shift_x
                      << " DY=" << left_raw_shift_y << "\n";
          }

          if (left_delta_x > left_deadzone || left_delta_y > left_deadzone) {
            last_shift_lx = left_raw_shift_x;
            last_shift_ly = left_raw_shift_y;
          }

          apply_spherical_warp(leftEyeRoi, left_map_x, left_map_y,
                               last_shift_lx, last_shift_ly, "LEFT");

          // --- Ojo Derecho: EMA + Deadzone ---
          float right_pupila_x_abs = roi.x + landmarks[96 * 2] * roi.width;
          float right_pupila_y_abs = roi.y + landmarks[96 * 2 + 1] * roi.height;
          float right_pupila_x_relativa =
              right_pupila_x_abs - static_cast<float>(rightEyeRoi.x);
          float right_pupila_y_relativa =
              right_pupila_y_abs - static_cast<float>(rightEyeRoi.y);
          float right_target_x = rightEyeRoi.width / 2.0f;
          float right_target_y = rightEyeRoi.height / 2.0f;
          float right_raw_shift_x = right_target_x - right_pupila_x_relativa;
          float right_raw_shift_y = right_target_y - right_pupila_y_relativa;

          float right_deadzone = 0.04f * static_cast<float>(rightEyeRoi.width);
          float right_delta_x = std::abs(right_raw_shift_x - last_shift_rx);
          float right_delta_y = std::abs(right_raw_shift_y - last_shift_ry);

          if (frameCounter % 30 == 0) {
            std::cerr
                << "ROI Der   : Width=" << rightEyeRoi.width
                << ", Height=" << rightEyeRoi.height << "\n"
                << "Target [R]: X=" << right_target_x << " Y=" << right_target_y
                << "\n"
                << "Actual [R]: X=" << right_pupila_x_relativa
                << " Y=" << right_pupila_y_relativa << "\n"
                << "Shift  [R]: DX=" << right_raw_shift_x
                << " DY=" << right_raw_shift_y << "\n"
                << "--------------------------------------------------------\n";
          }

          if (right_delta_x > right_deadzone ||
              right_delta_y > right_deadzone) {
            last_shift_rx = right_raw_shift_x;
            last_shift_ry = right_raw_shift_y;
          }

          apply_spherical_warp(rightEyeRoi, right_map_x, right_map_y,
                               last_shift_rx, last_shift_ry, "RIGHT");
        }

        // Edge-triggered logging: corrección activa
        bool warpFullyActive = (warpMultiplier >= 1.0f);
        if (warpFullyActive && !prevWarpFullyActive) {
          std::cout << "[DAEMON] Corrección de mirada activa\n";
        }
        prevWarpFullyActive = warpFullyActive;

        // Edge-triggered logging: parpadeo
        if (isBlinking && !prevBlinking) {
          std::cout << "[DAEMON] Parpadeo detectado (Pausa temporal)\n";
        }
        prevBlinking = isBlinking;

      } catch (const Ort::Exception &e) {
        std::cerr << "[Inferencia PFLD Error] " << e.what() << std::endl;
        return;
      }
    }
  } else if (effectEnabled) {
    if (prevFaceFound) {
      std::cout << "[DAEMON] Rostro perdido\n";
    }
  }

  prevFaceFound = faceFound;
}
