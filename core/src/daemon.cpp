#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir /dev/video0."
              << std::endl;
    return 1;
  }

  // Degradación grácil estricta y destrucción de buffer histórico
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  cap.set(cv::CAP_PROP_FPS, 15);
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  int fd = open("/dev/video2", O_RDWR);
  if (fd < 0) {
    std::cerr << "Falla de kernel: Descriptor /dev/video2 inaccesible."
              << std::endl;
    return 1;
  }

  // Negociación V4L2 estricta
  struct v4l2_format fmt = {0};
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = 640;
  fmt.fmt.pix.height = 360;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
  fmt.fmt.pix.sizeimage = 640 * 360 * 3;

  if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
    std::cerr
        << "Falla de I/O: Imposible negociar formato en descriptor virtual."
        << std::endl;
    return 1;
  }

  cv::Mat frame;
  const int benchmark_frames = 600;
  double total_latency = 0.0;

  std::cout << "[BENCHMARK] Productor activo. Ejecuta 'mpv' en la segunda "
               "terminal AHORA."
            << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    // 1. Espera de Hardware (Fuera del temporizador)
    cap.read(frame);
    if (frame.empty())
      break;

    // 2. Medición Estructural Pura (Zero-Copy)
    auto start = std::chrono::high_resolution_clock::now();

    write(fd, frame.data, frame.total() * frame.elemSize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    total_latency += elapsed.count();
  }

  std::cout << "Latencia promedio real: " << (total_latency / benchmark_frames)
            << " ms" << std::endl;

  close(fd);
  cap.release();
  return 0;
}