#pragma once

#include <opencv2/opencv.hpp>

struct EulerAnglesDeg {
  double pitch{0.0};
  double yaw{0.0};
  double roll{0.0};
};

class GeometryEngine {
public:
  cv::Mat renderEye(const cv::Mat &eyeCrop,
                    const EulerAnglesDeg &headPoseDeg) const;
  cv::Mat createBlendMask(const cv::Size &size) const;

private:
  cv::Matx33d buildRotationMatrix(const EulerAnglesDeg &headPoseDeg) const;
  cv::Vec3d liftToHemisphere(double nx, double ny) const;
  cv::Point2d projectOrthographic(const cv::Vec3d &point, double cx, double cy,
                                  double radius) const;
  cv::Vec3b sampleBilinear(const cv::Mat &src, double x, double y) const;
};
