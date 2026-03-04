// =============================================================================
// DirectLook — Client GUI (System Tray) — Multiplataforma
// =============================================================================

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
  // Stub: Client GUI pendiente de implementación
  (void)hInstance;
  (void)hPrevInstance;
  (void)lpCmdLine;
  (void)nCmdShow;
  return 0;
}

#else
// Linux: GTK3 stub
int main() { return 0; }

#endif