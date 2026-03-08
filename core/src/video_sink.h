#pragma once
#include <opencv2/opencv.hpp>

class VideoSink {
public:
  virtual ~VideoSink() = default;
  virtual void writeFrame(const cv::Mat &frame) = 0;
};
