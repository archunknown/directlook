// =============================================================================
// DirectLook — Client GUI (System Tray) — Multiplataforma
// =============================================================================

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <atomic>
#include <iostream>
#include <thread>

#define WM_TRAY_ICON (WM_USER + 1)
#define WM_THERMAL_ALARM (WM_USER + 2)
#define ID_TRAY_APP_ICON 1001
#define ID_TRAY_EXIT 1002
#define ID_TRAY_PAUSE 1003

std::atomic<bool> is_running(true);
std::atomic<bool> is_paused(false);
HANDLE g_hPipe = INVALID_HANDLE_VALUE;
HWND g_hwnd = NULL;
NOTIFYICONDATA g_nid = {};

// Hilo asíncrono para IPC
static void ipc_worker() {
  LPCSTR pipeName = "\\\\.\\pipe\\directlook_pipe";

  while (is_running) {
    g_hPipe =
        CreateFileA(pipeName, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (g_hPipe != INVALID_HANDLE_VALUE) {
      std::cout << "DirectLook [Client Win]: Connected to daemon Named Pipe."
                << std::endl;
      break;
    }

    // Wait for the pipe
    if (GetLastError() != ERROR_PIPE_BUSY) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      continue;
    }

    if (!WaitNamedPipeA(pipeName, 2000)) {
      continue;
    }
  }

  uint8_t buffer;
  DWORD bytesRead;

  while (is_running && g_hPipe != INVALID_HANDLE_VALUE) {
    bool result = ReadFile(g_hPipe, &buffer, 1, &bytesRead, NULL);

    if (result && bytesRead > 0) {
      if (buffer == 0x03) {
        // Notificación Thread-Safe a la UI principal
        if (g_hwnd) {
          PostMessage(g_hwnd, WM_THERMAL_ALARM, 0, 0);
        }
      }
    } else {
      // Error reading or pipe closed
      std::cout << "DirectLook [Client Win]: Daemon pipe disconnected or error."
                << std::endl;
      break;
    }
  }

  if (g_hPipe != INVALID_HANDLE_VALUE) {
    CloseHandle(g_hPipe);
  }
}

// Ventana invisible para procesar mensajes
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam,
                            LPARAM lParam) {
  switch (uMsg) {
  case WM_CREATE:
    g_nid.cbSize = sizeof(NOTIFYICONDATA);
    g_nid.hWnd = hwnd;
    g_nid.uID = ID_TRAY_APP_ICON;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY_ICON;
    g_nid.hIcon = LoadIcon(NULL, IDI_APPLICATION); // Default icon
    lstrcpyA(g_nid.szTip, "DirectLook");
    Shell_NotifyIcon(NIM_ADD, &g_nid);
    break;

  case WM_TRAY_ICON:
    if (lParam == WM_RBUTTONUP || lParam == WM_LBUTTONUP) {
      POINT pt;
      GetCursorPos(&pt);
      HMENU hMenu = CreatePopupMenu();

      UINT pauseFlags = MF_STRING;
      if (is_paused) {
        pauseFlags |= MF_CHECKED;
      }

      AppendMenuA(hMenu, pauseFlags, ID_TRAY_PAUSE, "Pausar Correccion");
      AppendMenuA(hMenu, MF_SEPARATOR, 0, NULL);
      AppendMenuA(hMenu, MF_STRING, ID_TRAY_EXIT, "Salir");

      SetForegroundWindow(hwnd);
      TrackPopupMenu(hMenu, TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, pt.x, pt.y, 0,
                     hwnd, NULL);
      DestroyMenu(hMenu);
    }
    break;

  case WM_COMMAND:
    switch (LOWORD(wParam)) {
    case ID_TRAY_PAUSE:
      is_paused = !is_paused;
      if (g_hPipe != INVALID_HANDLE_VALUE) {
        DWORD bytesWritten;
        uint8_t cmd = is_paused ? 0x01 : 0x00;
        WriteFile(g_hPipe, &cmd, 1, &bytesWritten, NULL);
      }
      std::cout << "DirectLook [Client Win]: Pause toggled to "
                << (is_paused ? "true" : "false") << std::endl;
      break;
    case ID_TRAY_EXIT:
      is_running = false;
      Shell_NotifyIcon(NIM_DELETE, &g_nid);
      PostQuitMessage(0);
      break;
    }
    break;

  case WM_THERMAL_ALARM:
    g_nid.uFlags = NIF_INFO;
    lstrcpyA(g_nid.szInfoTitle, "DirectLook Thermal Alarm");
    lstrcpyA(g_nid.szInfo,
             "Advertencia: Degradación visual por asfixia del hardware.");
    g_nid.dwInfoFlags = NIIF_WARNING;
    Shell_NotifyIcon(NIM_MODIFY, &g_nid);
    break;

  case WM_DESTROY:
    Shell_NotifyIcon(NIM_DELETE, &g_nid);
    PostQuitMessage(0);
    return 0;

  default:
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
  return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nCmdShow) {
  // Registro de clase de ventana
  const char CLASS_NAME[] = "DirectLookTrayClass";

  WNDCLASSEXA wc = {};
  wc.cbSize = sizeof(WNDCLASSEXA);
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = hInstance;
  wc.lpszClassName = CLASS_NAME;

  RegisterClassExA(&wc);

  // Creación de ventana invisible (Message-Only Window pattern pero con soporte
  // SysTray)
  g_hwnd = CreateWindowExA(0, CLASS_NAME, "DirectLook Logic Window", 0, 0, 0, 0,
                           0, NULL, NULL, hInstance, NULL);

  if (g_hwnd == NULL) {
    return 0;
  }

  // Levantamiento del hilo asíncrono
  std::thread ipc_thread(ipc_worker);
  ipc_thread.detach();

  // Bucle de mensajes estándar
  MSG msg = {};
  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

  is_running = false;
  return 0;
}

#else
// Linux: GTK3 with Ayatana AppIndicator and libnotify
#include <atomic>
#include <chrono>
#include <gtk/gtk.h>
#include <iostream>
#include <libayatana-appindicator/app-indicator.h>
#include <libnotify/notify.h>
#include <mutex>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

std::atomic<bool> is_running(true);
std::atomic<bool> is_paused(false);
int g_sock = -1;
std::mutex state_mutex;

static gboolean show_thermal_alarm_notification(gpointer data) {
  if (!notify_is_initted()) {
    notify_init("DirectLook");
  }

  NotifyNotification *notification = notify_notification_new(
      "DirectLook Thermal Alarm",
      "Advertencia: Degradación visual por asfixia del hardware.",
      "dialog-warning");
  notify_notification_set_urgency(notification, NOTIFY_URGENCY_CRITICAL);

  GError *error = nullptr;
  if (!notify_notification_show(notification, &error)) {
    std::cerr << "DirectLook [Client]: Error showing notification - "
              << error->message << std::endl;
    g_error_free(error);
  }
  g_object_unref(notification);

  return G_SOURCE_REMOVE;
}

static void ipc_worker() {
  g_sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (g_sock == -1) {
    std::cerr << "DirectLook [Client]: Failed to create IPC socket."
              << std::endl;
    return;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, "/tmp/directlook.sock", sizeof(addr.sun_path) - 1);

  while (is_running) {
    if (connect(g_sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      std::cout << "DirectLook [Client]: Connected to daemon IPC." << std::endl;
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  uint8_t buffer;
  while (is_running) {
    ssize_t bytes_read = read(g_sock, &buffer, 1);
    if (bytes_read > 0) {
      if (buffer == 0x03) {
        // Delegate notification rendering to the GTK main loop
        g_idle_add(show_thermal_alarm_notification, nullptr);
      }
    } else if (bytes_read == 0) {
      std::cout << "DirectLook [Client]: Daemon closed connection."
                << std::endl;
      break;
    } else {
      // Error doing read or socket closed abruptly
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }

  close(g_sock);
}

static void on_toggle_pause(GtkCheckMenuItem *item, gpointer data) {
  std::lock_guard<std::mutex> lock(state_mutex);
  is_paused = gtk_check_menu_item_get_active(item);

  if (g_sock != -1) {
    uint8_t cmd = is_paused ? 0x01 : 0x00;
    send(g_sock, &cmd, 1, MSG_NOSIGNAL);
  }

  std::cout << "DirectLook [Client]: Pause state toggled to "
            << (is_paused ? "true" : "false") << std::endl;
}

static void on_quit(GtkMenuItem *item, gpointer data) {
  is_running = false;
  gtk_main_quit();
}

int main(int argc, char **argv) {
  gtk_init(nullptr, nullptr); // Requiere inicialización gráfica pura
  notify_init("DirectLook");

  GtkWidget *menu = gtk_menu_new();

  GtkWidget *pause_item =
      gtk_check_menu_item_new_with_label("Pausar Corrección");
  g_signal_connect(pause_item, "toggled", G_CALLBACK(on_toggle_pause), nullptr);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), pause_item);

  GtkWidget *quit_item = gtk_menu_item_new_with_label("Salir");
  g_signal_connect(quit_item, "activate", G_CALLBACK(on_quit), nullptr);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), quit_item);

  gtk_widget_show_all(menu);

  AppIndicator *indicator =
      app_indicator_new("directlook-client",
                        "camera-web", // Icon name
                        APP_INDICATOR_CATEGORY_APPLICATION_STATUS);
  app_indicator_set_status(indicator, APP_INDICATOR_STATUS_ACTIVE);
  app_indicator_set_menu(indicator, GTK_MENU(menu));

  std::thread ipc_thread(ipc_worker);
  ipc_thread.detach();

  gtk_main();

  is_running = false;
  notify_uninit();
  return 0;
}
#endif