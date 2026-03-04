#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
// ==========================================
// ARQUITECTURA WINDOWS (DirectShow / MSMF)
// ==========================================
#include <windows.h>

int main() {
  // En Windows, '0' selecciona la cámara por defecto usando el backend nativo
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir cámara en Windows."
              << std::endl;
    return 1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
  cap.set(cv::CAP_PROP_FPS, 15);

  cv::Mat frame;
  const int benchmark_frames = 600;
  double total_latency = 0.0;

  std::cout << "[BENCHMARK] Productor activo en Windows. Procesando frames..."
            << std::endl;

  for (int i = 0; i < benchmark_frames; ++i) {
    cap.read(frame);
    if (frame.empty())
      break;

    auto start = std::chrono::high_resolution_clock::now();

    // Para Windows, de momento abrimos una ventana de debug en lugar de
    // inyectar a una cámara virtual del kernel de Linux
    cv::imshow("DirectLook - Debug Windows", frame);
    cv::waitKey(1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    total_latency += elapsed.count();
  }

  std::cout << "Latencia promedio real: " << (total_latency / benchmark_frames)
            << " ms" << std::endl;

  cap.release();
  cv::destroyAllWindows();
  return 0;
}

#else
// ==========================================
// ARQUITECTURA LINUX (v4l2loopback nativo)
// ==========================================
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
  cv::VideoCapture cap(0, cv::CAP_V4L2);
  if (!cap.isOpened()) {
    std::cerr << "Falla estructural: Imposible adquirir /dev/video0."
              << std::endl;
    return 1;
  }

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
    cap.read(frame);
    if (frame.empty())
      break;

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
#endif