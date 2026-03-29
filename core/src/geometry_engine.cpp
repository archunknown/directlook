#include "geometry_engine.hpp"

#include <algorithm>

cv::Mat GeometryEngine::applyDisplacement(const cv::Mat &eyeCrop,
                                          const std::vector<cv::Point> &lidPts,
                                          float dx, float dy) const {
  // Validate inputs.
  if (eyeCrop.empty() || eyeCrop.cols != 64 || eyeCrop.rows != 64 ||
      eyeCrop.channels() != 3 || lidPts.size() < 3) {
    return cv::Mat();
  }

  // Clamp displacement to prevent remap artifacts at extreme angles.
  constexpr float kMaxDisp = 12.0f;
  dx = std::max(-kMaxDisp, std::min(kMaxDisp, dx));
  dy = std::max(-kMaxDisp, std::min(kMaxDisp, dy));

  // -----------------------------------------------------------------------
  // Step a — binary eyelid mask (CV_8UC1, 64x64).
  // Pixels inside the polygon = 255, outside = 0.
  // -----------------------------------------------------------------------
  cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
  {
    // Clamp all points to [0,63] so fillPoly never reads/writes out-of-bounds.
    std::vector<cv::Point> clamped;
    clamped.reserve(lidPts.size());
    for (const auto &p : lidPts) {
      clamped.push_back(
          {std::max(0, std::min(63, p.x)), std::max(0, std::min(63, p.y))});
    }
    const std::vector<std::vector<cv::Point>> contours = {clamped};
    cv::fillPoly(mask, contours, cv::Scalar(255));
  }

  // Degenerate polygon guard: if the mask is all-zero (collinear landmarks,
  // width-1 polygon, etc.) return unchanged crop — no-op is correct behavior.
  if (cv::countNonZero(mask) == 0) {
    return eyeCrop.clone();
  }

  // -----------------------------------------------------------------------
  // Step b — Euclidean distance field (CV_32FC1, 64x64).
  // Value at each pixel = distance-to-nearest-zero (boundary or exterior).
  // -----------------------------------------------------------------------
  cv::Mat dist32f;
  cv::distanceTransform(mask, dist32f, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  // -----------------------------------------------------------------------
  // Step c — normalize to [0.0, 1.0] using only interior pixels as reference.
  // weight == 1.0 at the innermost pixel, 0.0 at boundary and outside.
  // -----------------------------------------------------------------------
  cv::Mat weight;
  cv::normalize(dist32f, weight, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1, mask);

  // -----------------------------------------------------------------------
  // Step c2 — radial attenuation: pixels near the crop centre displace more
  // than those at the iris edge, preserving the circular iris shape.
  // Quadratic falloff: 1.0 at (32,32), 0.0 at maxRadius = 28px.
  // Static grid: computed once, reused every call.
  // -----------------------------------------------------------------------
  // Atenuación radial suave: coseno, no cuadrática
  // Mantiene fuerza alta en la zona central (iris) y cae gradualmente
  static cv::Mat radialWeight = []() {
    cv::Mat rw(64, 64, CV_32FC1);
    const float rcx = 32.0f, rcy = 32.0f;
    const float maxRadius = 30.0f;
    for (int r = 0; r < 64; ++r) {
      float *row = rw.ptr<float>(r);
      for (int c = 0; c < 64; ++c) {
        float dist = std::sqrt((c - rcx) * (c - rcx) + (r - rcy) * (r - rcy));
        float t = std::min(dist / maxRadius, 1.0f);
        // Coseno: 1.0 en el centro, cae suavemente a 0.0
        row[c] = 0.5f * (1.0f + std::cos(t * 3.14159265f));
      }
    }
    return rw;
  }();

  cv::multiply(weight, radialWeight, weight);

  // -----------------------------------------------------------------------
  // Step d — build remap coordinate grids.
  // map_x(y,x) = x - dx * weight(y,x)
  // map_y(y,x) = y - dy * weight(y,x)
  //
  // Grids are static to avoid per-frame heap allocation; safe in C++11+
  // (static local init is thread-safe) and in the current single-threaded
  // process() call path.
  // -----------------------------------------------------------------------
  static cv::Mat gridX = []() {
    cv::Mat g(64, 64, CV_32FC1);
    for (int r = 0; r < 64; ++r)
      for (int c = 0; c < 64; ++c)
        g.at<float>(r, c) = static_cast<float>(c);
    return g;
  }();

  static cv::Mat gridY = []() {
    cv::Mat g(64, 64, CV_32FC1);
    for (int r = 0; r < 64; ++r)
      for (int c = 0; c < 64; ++c)
        g.at<float>(r, c) = static_cast<float>(r);
    return g;
  }();

  cv::Mat map_x = gridX - dx * weight;
  cv::Mat map_y = gridY - dy * weight;

  // -----------------------------------------------------------------------
  // Step e — remap.
  // INTER_LINEAR matches the quality of the removed sampleBilinear helper.
  // BORDER_REFLECT_101 prevents black seams at edges without repeating the
  // edge pixel itself; handles up to ~12 px overshoot cleanly.
  // -----------------------------------------------------------------------
  cv::Mat remapped;
  cv::remap(eyeCrop, remapped, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_REFLECT_101);

  // -----------------------------------------------------------------------
  // Step f — masked copy-back.
  // Start from a clone of the original crop, then overwrite only the pixels
  // inside the eyelid polygon with the remapped result.
  // Pixels outside the polygon are byte-for-byte identical to eyeCrop.
  // -----------------------------------------------------------------------
  cv::Mat result = eyeCrop.clone();
  remapped.copyTo(result, mask);

  return result;
}

