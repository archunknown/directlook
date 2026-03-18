#include "video_sink_unix.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

LinuxV4l2Sink::LinuxV4l2Sink(const std::string &devicePath) : fd(-1) {
  fd = open(devicePath.c_str(), O_WRONLY);
  if (fd < 0) {
    throw std::runtime_error("Falla de kernel: Descriptor " + devicePath +
                             " inaccesible.");
  }

  std::cerr << "\n=== INICIO AUDITORIA V4L2 (" << devicePath
            << ") ===" << std::endl;

  struct v4l2_capability cap;
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
    std::cerr << "[CAP] Driver: " << cap.driver << " | Card: " << cap.card
              << std::endl;
    std::cerr << "[CAP] Capabilities: 0x" << std::hex << cap.capabilities
              << std::dec << std::endl;
  } else {
    std::cerr << "[CAP] Falla al consultar capacidades: "
              << std::strerror(errno) << std::endl;
  }

  struct v4l2_fmtdesc fmtdesc;
  std::memset(&fmtdesc, 0, sizeof(fmtdesc));
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  std::cerr << "[FMT] Formatos de OUTPUT admitidos por el kernel:" << std::endl;

  bool found_formats = false;
  while (ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
    found_formats = true;
    std::cerr << "  -> " << fmtdesc.description << " ["
              << (char)(fmtdesc.pixelformat & 0xFF)
              << (char)((fmtdesc.pixelformat >> 8) & 0xFF)
              << (char)((fmtdesc.pixelformat >> 16) & 0xFF)
              << (char)((fmtdesc.pixelformat >> 24) & 0xFF) << "]" << std::endl;
    fmtdesc.index++;
  }
  if (!found_formats) {
    std::cerr << "  -> NINGUNO. El dispositivo rechaza operaciones de OUTPUT."
              << std::endl;
  }
  std::cerr << "=== FIN AUDITORIA V4L2 ===\n" << std::endl;

  struct v4l2_format fmt;
  std::memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = 640;
  fmt.fmt.pix.height = 360;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  fmt.fmt.pix.bytesperline = 0;
  fmt.fmt.pix.sizeimage = 0;
  fmt.fmt.pix.colorspace = 0;

  std::cerr << "[NEGOCIACION] Forzando inyeccion: YUYV 640x360" << std::endl;
  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    std::string error_desc = std::strerror(errno);
    close(fd);
    throw std::runtime_error("Falla de I/O en VIDIOC_S_FMT: " + error_desc);
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
    cv::Mat processed_frame = frame;

    if (processed_frame.cols != 640 || processed_frame.rows != 360) {
      cv::resize(processed_frame, processed_frame, cv::Size(640, 360));
    }

    cv::Mat yuv_frame;
    cv::cvtColor(processed_frame, yuv_frame, cv::COLOR_BGR2YUV_YUYV);

    ssize_t written =
        write(fd, yuv_frame.data, yuv_frame.total() * yuv_frame.elemSize());
    if (written < 0) {
      std::cerr << "[SINK] Falla al escribir en dispositivo virtual. Código de "
                   "error: "
                << std::strerror(errno) << std::endl;
    }
  }
}