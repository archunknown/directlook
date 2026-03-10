#include "video_sink_windows.h"
#include <iostream>
#include <stdexcept>

WindowsVirtualCamSink::WindowsVirtualCamSink()
    : hMapFile(nullptr), hMutex(nullptr), pBuf(nullptr),
      bufferSize(640 * 360 * 3) {

  hMutex = CreateMutexA(NULL, FALSE, "Local\\DirectLookVirtualCam_Mutex");
  if (hMutex == NULL) {
    throw std::runtime_error("Falla de concurrencia: No se pudo crear el Mutex "
                             "para memoria compartida (Error " +
                             std::to_string(GetLastError()) + ").");
  }

  hMapFile = CreateFileMappingA(
      INVALID_HANDLE_VALUE,           // Use paging file
      NULL,                           // Default security
      PAGE_READWRITE,                 // Read/write access
      0,                              // Maximum object size (high-order DWORD)
      static_cast<DWORD>(bufferSize), // Maximum object size (low-order DWORD)
      "Local\\DirectLookVirtualCam"); // Name of mapping object

  if (hMapFile == NULL) {
    CloseHandle(hMutex);
    throw std::runtime_error(
        "Falla de memoria: CreateFileMappingA falló (Error " +
        std::to_string(GetLastError()) + ").");
  }

  pBuf = MapViewOfFile(hMapFile,            // Handle to map object
                       FILE_MAP_ALL_ACCESS, // Read/write permission
                       0, 0, bufferSize);

  if (pBuf == NULL) {
    CloseHandle(hMapFile);
    CloseHandle(hMutex);
    throw std::runtime_error("Falla de I/O: MapViewOfFile falló (Error " +
                             std::to_string(GetLastError()) + ").");
  }

  std::cout << "[SINK] Memoria Compartida iniciada: Local\\DirectLookVirtualCam"
            << std::endl;
}

WindowsVirtualCamSink::~WindowsVirtualCamSink() {
  if (pBuf != NULL) {
    UnmapViewOfFile(pBuf);
  }
  if (hMapFile != NULL) {
    CloseHandle(hMapFile);
  }
  if (hMutex != NULL) {
    CloseHandle(hMutex);
  }
}

void WindowsVirtualCamSink::writeFrame(const cv::Mat &frame) {
  if (pBuf == NULL || frame.empty() ||
      frame.total() * frame.elemSize() != bufferSize) {
    return;
  }

  // Adquirir bloqueo exclusivo
  DWORD dwWaitResult = WaitForSingleObject(
      hMutex, 50); // Timeout 50ms para no bloquear el daemon indefinidamente en
                   // caso de stall

  if (dwWaitResult == WAIT_OBJECT_0 || dwWaitResult == WAIT_ABANDONED) {
    std::memcpy(pBuf, frame.data, bufferSize);
    ReleaseMutex(hMutex);
  } else {
    std::cerr << "[SINK] Mutex Timeout: Tearing prevenido (Frame Drop)."
              << std::endl;
  }
}
