#include "cpu_monitor.h"
#include <chrono>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fstream>
#include <sstream>
#include <string>
#endif

CpuMonitor::CpuMonitor() {
  // Inicialización de la primera lectura base
  getCpuUsage();
  workerThread = std::thread(&CpuMonitor::monitorLoop, this);
}

CpuMonitor::~CpuMonitor() {
  keepRunning.store(false);
  if (workerThread.joinable()) {
    workerThread.join();
  }
}

int CpuMonitor::getDegradationLevel() const {
  return degradationLevel.load();
}

#ifdef _WIN32
unsigned long long CpuMonitor::fileTimeToULL(const void *ft) const {
  const FILETIME *ftp = static_cast<const FILETIME *>(ft);
  ULARGE_INTEGER uli;
  uli.LowPart = ftp->dwLowDateTime;
  uli.HighPart = ftp->dwHighDateTime;
  return uli.QuadPart;
}
#endif

double CpuMonitor::getCpuUsage() {
#ifdef _WIN32
  FILETIME idleTime, kernelTime, userTime;
  if (!GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
    return 0.0;
  }

  unsigned long long currentIdleTime = fileTimeToULL(&idleTime);
  unsigned long long currentKernelTime = fileTimeToULL(&kernelTime);
  unsigned long long currentUserTime = fileTimeToULL(&userTime);

  unsigned long long idleDiff = currentIdleTime - lastIdleTime;
  unsigned long long kernelDiff = currentKernelTime - lastKernelTime;
  unsigned long long userDiff = currentUserTime - lastUserTime;

  unsigned long long systemDiff = kernelDiff + userDiff;
  
  double cpu = 0.0;
  if (systemDiff > 0) {
    cpu = (systemDiff - idleDiff) * 100.0 / systemDiff;
  }

  lastIdleTime = currentIdleTime;
  lastKernelTime = currentKernelTime;
  lastUserTime = currentUserTime;

  return cpu;
#else
  std::ifstream statFile("/proc/stat");
  if (!statFile.is_open()) return 0.0;
  
  std::string line;
  std::getline(statFile, line);
  statFile.close();

  std::istringstream iss(line);
  std::string cpuStr;
  iss >> cpuStr;

  unsigned long long user, nice, sys, idle;
  iss >> user >> nice >> sys >> idle;

  double percent = 0.0;
  if (lastTotalUser > 0 || lastTotalUserLow > 0 || lastTotalSys > 0 || lastTotalIdle > 0) {
    unsigned long long total = (user - lastTotalUser) + (nice - lastTotalUserLow) + (sys - lastTotalSys);
    percent = total * 100.0 / (total + (idle - lastTotalIdle));
  }

  lastTotalUser = user;
  lastTotalUserLow = nice;
  lastTotalSys = sys;
  lastTotalIdle = idle;

  return percent;
#endif
}

void CpuMonitor::monitorLoop() {
  while (keepRunning.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    double currentUsage = getCpuUsage();

    if (currentUsage > 95.0) {
      level3Cycles++;
      level2Cycles = 0;
      level1Cycles = 0;
      recoveryCycles = 0;
      if (level3Cycles >= 10) {
        degradationLevel.store(3);
        level3Cycles = 10;
      }
    } else if (currentUsage > 85.0) {
      level3Cycles = 0;
      level2Cycles++;
      level1Cycles = 0;
      recoveryCycles = 0;
      if (level2Cycles >= 10) {
        degradationLevel.store(2);
        level2Cycles = 10;
      }
    } else if (currentUsage > 70.0) {
      level3Cycles = 0;
      level2Cycles = 0;
      level1Cycles++;
      recoveryCycles = 0;
      if (level1Cycles >= 10) {
        degradationLevel.store(1);
        level1Cycles = 10;
      }
    } else if (currentUsage < 60.0) {
      level3Cycles = 0;
      level2Cycles = 0;
      level1Cycles = 0;
      recoveryCycles++;
      if (recoveryCycles >= 10) {
        degradationLevel.store(0);
        recoveryCycles = 10;
      }
    }
  }
}
