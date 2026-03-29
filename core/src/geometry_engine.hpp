#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class GeometryEngine {
public:
  // Applies a 2D displacement field to eyeCrop, weighted by the signed
  // distance field of the eyelid polygon.
  //
  // eyeCrop  – CV_8UC3, exactly 64x64 pixels, cloned from frame.
  // lidPts   – 8 points in local ROI coordinates [0..63, 0..63],
  //            WFLW-98 indices 60-67 (left eye) or 68-75 (right eye).
  // dx, dy   – displacement in pixels; positive dx shifts source sampling
  //            rightward (apparent gaze left). Clamped internally to [-12,12].
  //
  // Returns CV_8UC3 64x64. Only pixels inside the eyelid polygon are
  // displaced; exterior pixels are copied unchanged from eyeCrop.
  // Returns an empty Mat on any validation failure.
  cv::Mat applyDisplacement(const cv::Mat &eyeCrop,
                            const std::vector<cv::Point> &lidPts,
                            float dx, float dy) const;
};
