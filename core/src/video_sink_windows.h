#pragma once
#include "video_sink.h"
#include <windows.h>

class WindowsVirtualCamSink : public VideoSink {
public:
  WindowsVirtualCamSink();
  ~WindowsVirtualCamSink() override;

  WindowsVirtualCamSink(const WindowsVirtualCamSink &) = delete;
  WindowsVirtualCamSink &operator=(const WindowsVirtualCamSink &) = delete;

  void writeFrame(const cv::Mat &frame) override;

private:
  HANDLE hMapFile;
  HANDLE hMutex;
  void *pBuf;
  size_t bufferSize;
};
