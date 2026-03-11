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
  
#ifdef _WIN32
  SetThreadPriority(workerThread.native_handle(), THREAD_PRIORITY_LOWEST);
#else
  sched_param param;
  param.sched_priority = 0;
  pthread_setschedparam(workerThread.native_handle(), SCHED_IDLE, &param);
#endif
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

#ifdef _WIN32
double CpuMonitor::getCpuUsage() {
  FILETIME creationTime, exitTime, kernelTime, userTime;
  FILETIME sysIdleTime, sysKernelTime, sysUserTime;

  if (!GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime)) {
    return 0.0;
  }
  if (!GetSystemTimes(&sysIdleTime, &sysKernelTime, &sysUserTime)) {
    return 0.0;
  }

  unsigned long long currentProcessTime = fileTimeToULL(&kernelTime) + fileTimeToULL(&userTime);
  unsigned long long currentSystemTime = fileTimeToULL(&sysKernelTime) + fileTimeToULL(&sysUserTime);

  double cpu = 0.0;
  if (lastSystemTime > 0 || lastProcessTime > 0) {
    unsigned long long processDiff = currentProcessTime - lastProcessTime;
    unsigned long long systemDiff = currentSystemTime - lastSystemTime;
    if (systemDiff > 0) {
      cpu = (processDiff * 100.0) / systemDiff;
    }
  }

  lastProcessTime = currentProcessTime;
  lastSystemTime = currentSystemTime;

  return cpu;
}
#else
double CpuMonitor::getCpuUsage() {
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
  unsigned long long currentSysTotal = user + nice + sys + idle;

  std::ifstream procFile("/proc/self/stat");
  if (!procFile.is_open()) return 0.0;
  
  std::string procLine;
  std::getline(procFile, procLine);
  procFile.close();

  std::istringstream procIss(procLine);
  std::string token;
  unsigned long long utime = 0, stime = 0;
  for (int i = 1; i <= 15; ++i) {
    procIss >> token;
    if (i == 14) utime = std::stoull(token);
    if (i == 15) stime = std::stoull(token);
  }
  unsigned long long currentProcTotal = utime + stime;

  double percent = 0.0;
  if (lastSysTotal > 0 || lastProcTotal > 0) {
    unsigned long long procDiff = currentProcTotal - lastProcTotal;
    unsigned long long sysDiff = currentSysTotal - lastSysTotal;
    if (sysDiff > 0) {
      percent = (procDiff * 100.0) / sysDiff;
    }
  }

  lastSysTotal = currentSysTotal;
  lastProcTotal = currentProcTotal;

  return percent;
  return percent;
}
#endif

void CpuMonitor::monitorLoop() {
  while (keepRunning.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    double currentUsage = getCpuUsage();

    if (currentUsage > 70.0) {
      escalationCycles++;
      recoveryCycles = 0;
      if (escalationCycles >= 10) {
        int currentLevel = degradationLevel.load();
        if (currentLevel < 3) {
          degradationLevel.store(currentLevel + 1);
        }
        escalationCycles = 0; // Reset para los siguientes 5 segundos
      }
    } else if (currentUsage < 20.0) {
      recoveryCycles++;
      escalationCycles = 0;
      if (recoveryCycles >= 30) {
        int currentLevel = degradationLevel.load();
        if (currentLevel > 0) {
          degradationLevel.store(currentLevel - 1);
        }
        recoveryCycles = 0;
      }
    } else {
      // Entre 50% y 70%, se mantiene el estado estable
      escalationCycles = 0;
      recoveryCycles = 0;
    }
  }
}
