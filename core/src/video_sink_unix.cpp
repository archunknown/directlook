#include "video_sink_unix.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

LinuxV4l2Sink::LinuxV4l2Sink(const std::string &devicePath) : fd(-1) {
  fd = open(devicePath.c_str(), O_RDWR);
  if (fd < 0) {
    throw std::runtime_error("Falla de kernel: Descriptor " + devicePath +
                             " inaccesible.");
  }

  struct v4l2_format fmt;
  std::memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = 640;
  fmt.fmt.pix.height = 360;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
  fmt.fmt.pix.sizeimage = 640 * 360 * 3;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    close(fd);
    throw std::runtime_error(
        "Falla de I/O: Imposible negociar formato en descriptor virtual.");
  }
  std::cout << "[SINK] Loopback virtual activado en " << devicePath
            << std::endl;
}

LinuxV4l2Sink::~LinuxV4l2Sink() {
  if (fd >= 0) {
    close(fd);
  }
}

void LinuxV4l2Sink::writeFrame(const cv::Mat &frame) {
  if (fd >= 0 && !frame.empty()) {
    ssize_t written = write(fd, frame.data, frame.total() * frame.elemSize());
    if (written < 0) {
      std::cerr << "[SINK] Falla al escribir en dispositivo virtual."
                << std::endl;
    }
  }
}
