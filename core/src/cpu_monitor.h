#pragma once
#include <atomic>
#include <thread>

class CpuMonitor {
public:
  CpuMonitor();
  ~CpuMonitor();

  // Devuelve el nivel estrangulamiento térmico (0, 1, 2 o 3)
  int getDegradationLevel() const;

private:
  void monitorLoop();
  double getCpuUsage();

#ifdef _WIN32
  // Helpers Windows para cálculos de tiempo
  unsigned long long fileTimeToULL(const void *ft) const;
  unsigned long long lastIdleTime = 0;
  unsigned long long lastKernelTime = 0;
  unsigned long long lastUserTime = 0;
#else
  // Helpers Linux para cálculos de tiempo
  unsigned long long lastTotalUser = 0;
  unsigned long long lastTotalUserLow = 0;
  unsigned long long lastTotalSys = 0;
  unsigned long long lastTotalIdle = 0;
#endif

  std::atomic<bool> keepRunning{true};
  std::atomic<int> degradationLevel{0};
  std::thread workerThread;

  int escalationCycles{0};
  int recoveryCycles{0};
};
