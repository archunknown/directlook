#include "geometry_engine.hpp"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

cv::Mat GeometryEngine::renderEye(const cv::Mat &eyeCrop,
                                  const EulerAnglesDeg &headPoseDeg) const {
  if (eyeCrop.empty() || eyeCrop.channels() != 3) {
    return cv::Mat();
  }

  cv::Mat rendered = eyeCrop.clone();
  const cv::Matx33d rotation = buildRotationMatrix(headPoseDeg);
  const double centerX = (eyeCrop.cols - 1) * 0.5;
  const double centerY = (eyeCrop.rows - 1) * 0.5;
  const double radius = 0.5 * static_cast<double>(std::min(eyeCrop.cols, eyeCrop.rows));

  for (int y = 0; y < eyeCrop.rows; ++y) {
    for (int x = 0; x < eyeCrop.cols; ++x) {
      const double nx = (x - centerX) / radius;
      const double ny = (y - centerY) / radius;
      if (nx * nx + ny * ny > 1.0) {
        continue;
      }

      const cv::Vec3d dstPoint = liftToHemisphere(nx, ny);
      const cv::Vec3d srcPoint = rotation * dstPoint;
      if (srcPoint[2] <= 0.0) {
        continue;
      }

      const cv::Point2d srcUv =
          projectOrthographic(srcPoint, centerX, centerY, radius);
      if (srcUv.x < 0.0 || srcUv.y < 0.0 || srcUv.x >= eyeCrop.cols - 1 ||
          srcUv.y >= eyeCrop.rows - 1) {
        continue;
      }

      rendered.at<cv::Vec3b>(y, x) = sampleBilinear(eyeCrop, srcUv.x, srcUv.y);
    }
  }

  return rendered;
}

cv::Mat GeometryEngine::createBlendMask(const cv::Size &size) const {
  if (size.width <= 0 || size.height <= 0) {
    return cv::Mat();
  }

  cv::Mat mask(size, CV_8UC1, cv::Scalar(0));
  const int radius = std::max(1, std::min(size.width, size.height) / 2 - 2);
  const cv::Point center(size.width / 2, size.height / 2);
  cv::circle(mask, center, radius, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
  return mask;
}

cv::Matx33d
GeometryEngine::buildRotationMatrix(const EulerAnglesDeg &headPoseDeg) const {
  const double pitch = headPoseDeg.pitch * kPi / 180.0;
  const double yaw = headPoseDeg.yaw * kPi / 180.0;
  const double roll = headPoseDeg.roll * kPi / 180.0;

  const cv::Matx33d rx(1.0, 0.0, 0.0, 0.0, std::cos(pitch), -std::sin(pitch),
                       0.0, std::sin(pitch), std::cos(pitch));
  const cv::Matx33d ry(std::cos(yaw), 0.0, std::sin(yaw), 0.0, 1.0, 0.0,
                       -std::sin(yaw), 0.0, std::cos(yaw));
  const cv::Matx33d rz(std::cos(roll), -std::sin(roll), 0.0, std::sin(roll),
                       std::cos(roll), 0.0, 0.0, 0.0, 1.0);

  return rz * ry * rx;
}

cv::Vec3d GeometryEngine::liftToHemisphere(double nx, double ny) const {
  const double radial = std::max(0.0, 1.0 - nx * nx - ny * ny);
  return cv::Vec3d(nx, ny, std::sqrt(radial));
}

cv::Point2d GeometryEngine::projectOrthographic(const cv::Vec3d &point,
                                                double cx, double cy,
                                                double radius) const {
  return cv::Point2d(cx + radius * point[0], cy + radius * point[1]);
}

cv::Vec3b GeometryEngine::sampleBilinear(const cv::Mat &src, double x,
                                         double y) const {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, src.cols - 1);
  const int y1 = std::min(y0 + 1, src.rows - 1);
  const double tx = x - x0;
  const double ty = y - y0;

  const cv::Vec3d top =
      cv::Vec3d(src.at<cv::Vec3b>(y0, x0)) * (1.0 - tx) +
      cv::Vec3d(src.at<cv::Vec3b>(y0, x1)) * tx;
  const cv::Vec3d bottom =
      cv::Vec3d(src.at<cv::Vec3b>(y1, x0)) * (1.0 - tx) +
      cv::Vec3d(src.at<cv::Vec3b>(y1, x1)) * tx;
  const cv::Vec3d value = top * (1.0 - ty) + bottom * ty;

  return cv::Vec3b(cv::saturate_cast<uchar>(value[0]),
                   cv::saturate_cast<uchar>(value[1]),
                   cv::saturate_cast<uchar>(value[2]));
}
