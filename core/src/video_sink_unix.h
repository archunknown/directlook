#pragma once
#include "video_sink.h"
#include <string>

class LinuxV4l2Sink : public VideoSink {
public:
  explicit LinuxV4l2Sink(const std::string &devicePath);
  ~LinuxV4l2Sink() override;

  LinuxV4l2Sink(const LinuxV4l2Sink &) = delete;
  LinuxV4l2Sink &operator=(const LinuxV4l2Sink &) = delete;

  void writeFrame(const cv::Mat &frame) override;

private:
  int fd;
};
