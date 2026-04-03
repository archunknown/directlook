<div align="center">

# DirectLook

### Corrección de Mirada en Tiempo Real

**Herramienta nativa para Windows y Linux que simula contacto visual directo mediante una red neuronal ligera, sin requerir GPU dedicada.**

[![C++17](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)]()
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24.2-orange.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()

---

</div>

## Qué es DirectLook

DirectLook intercepta el flujo de video de tu cámara, detecta tus ojos en cada fotograma, y redirige la mirada para que parezca que miras directamente a la cámara — incluso cuando estás leyendo la pantalla, revisando notas o mirando un segundo monitor.

El resultado se expone como una **cámara virtual** que cualquier aplicación de videoconferencia (Zoom, Meet, Teams) puede seleccionar.

### Comparación con NVIDIA Eye Contact

| | NVIDIA Eye Contact | DirectLook |
|---|---|---|
| **Arquitectura** | GAN generativa | U-Net residual + Pipeline C++ |
| **Hardware requerido** | GPU NVIDIA con Tensor Cores | Cualquier CPU con AVX2 |
| **Peso del modelo** | >150 MB | ~4.3 MB (FP32) |
| **RAM** | 1.5 – 2 GB | <110 MB (daemon) + <35 MB (GUI) |
| **Latencia** | 150 – 300 ms | 22 – 30 ms |

---

## Arquitectura

La arquitectura se divide en **dos procesos aislados**. El aislamiento es innegociable: si la GUI falla, el video no se interrumpe.

```
┌─────────────────────────────┐     IPC      ┌──────────────────────┐
│     Motor (Daemon Core)     │◄────────────►│  Cliente (System Tray) │
│                             │  Named Pipes  │                      │
│  • Captura de video         │  / Unix Socket│  • Activar/Desactivar│
│  • Detección facial         │               │  • Alertas térmicas  │
│  • Landmarks (PFLD)         │               │  • <35 MB RAM        │
│  • Corrección NN (U-Net)    │               └──────────────────────┘
│  • Cámara virtual           │
│  • <125 MB RAM, <30ms       │
└─────────────────────────────┘
```

### Pipeline de Procesamiento (por fotograma)

| Etapa | Modelo/Operación | Latencia |
|---|---|---|
| 1. Detección facial | UltraFace-slim-320 | 3.5 – 5 ms |
| 2. Landmarks faciales | PFLD-68 (98 pts, 68 válidos) | 17 – 19 ms |
| 3. Extracción ROI ocular | Bounding box 64×64 + padding 20% | <0.1 ms |
| 4. Normalización canónica | Transformación afín (centrado, rotación, escala) | <0.1 ms |
| 5. Corrección de mirada | GazeCorrector U-Net residual | ~6 ms (2 ojos) |
| 6. Transformación inversa | cv::invertAffineTransform | <0.1 ms |
| 7. Post-procesamiento | Normalización de color + alpha blending | <0.5 ms |
| **Total** | | **~28 ms** |

---

## Requisitos

### Hardware mínimo

- **CPU:** Intel Core i5 8.ª gen / AMD Ryzen 5 serie 3000 (4 núcleos, AVX2, >3.4 GHz turbo)
- **RAM:** 8 GB DDR4
- **GPU:** No requerida

### Software

| | Windows | Linux |
|---|---|---|
| **Compilador** | MSVC (VS 2019+) | GCC / Clang |
| **Dependencias** | OpenCV 4.x, ONNX Runtime 1.24+ | OpenCV 4.x, ONNX Runtime 1.16+ |
| **GUI** | Win32 API | GTK3 + Ayatana AppIndicator |
| **IPC** | Named Pipes | Unix Domain Sockets |
| **Cámara Virtual** | Shared Memory + DirectShow | v4l2loopback |
| **Build System** | CMake 3.15+ | CMake 3.15+ |

---

## Compilación

### Windows

Usar **"x64 Native Tools Command Prompt for VS"** (no la terminal genérica).

```cmd
cd C:\Users\<tu_usuario>\directlook

:: Configurar (primera vez o después de limpiar)
cmake -S . -B build -A x64 -DOpenCV_RUNTIME=vc16

:: Compilar
cmake --build build --config Release

:: Copiar DLL de ONNX Runtime (necesario tras build limpio)
copy models\onnxruntime-win-x64-1.24.2\lib\onnxruntime.dll build\core\Release\
```

### Linux

```bash
cd ~/directlook
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

---

## Ejecución

### Daemon (Motor)

```bash
# Con webcam (Windows)
build\core\Release\directlook-daemon.exe --camera 0

# Con video de prueba (fallback)
build\core\Release\directlook-daemon.exe

# Linux con cámara virtual
./build/core/directlook-daemon --camera 0 --video-sink /dev/video2
```

### Cliente GUI

```bash
# Windows
build\client\Release\directlook-client.exe

# Linux
./build/client/directlook-client &
```

El icono aparece en la bandeja del sistema. Click derecho para pausar la corrección o salir.

---

## Estructura del Proyecto

```
directlook/
├── core/                       # Motor (Daemon)
│   ├── src/
│   │   ├── daemon.cpp          # Entry point + main loop
│   │   ├── vision_pipeline.cpp # Pipeline completo de visión
│   │   ├── vision_pipeline.h
│   │   ├── geometry_engine.*   # Motor geométrico (legacy, sin uso)
│   │   ├── temporal_filter.*   # One Euro Filter para landmarks
│   │   ├── cpu_monitor.*       # Monitor térmico
│   │   ├── ipc_server.h        # Interfaz IPC abstracta
│   │   ├── ipc_windows.*       # Named Pipes (Windows)
│   │   ├── ipc_unix.*          # Unix Sockets (Linux)
│   │   ├── video_sink.h        # Interfaz de salida abstracta
│   │   ├── video_sink_windows.*# Shared Memory (Windows)
│   │   ├── video_sink_unix.*   # v4l2loopback (Linux)
│   │   └── protocol.h          # Protocolo IPC (1 byte)
│   └── CMakeLists.txt
├── client/                     # Cliente GUI
│   ├── src/tray.cpp            # Win32 + GTK3 multiplataforma
│   └── CMakeLists.txt
├── modelos/                    # Modelos ONNX (ignorados por .gitignore)
│   ├── version-slim-320_simplified.onnx  # UltraFace
│   ├── pfld.onnx                         # PFLD landmarks
│   ├── iris_landmark.onnx                # MediaPipe Iris (diagnóstico)
│   └── gaze_corrector.onnx              # U-Net corrección de mirada
├── models/
│   └── onnxruntime-win-x64-1.24.2/      # SDK ONNX Runtime
├── extract_training_pairs.py   # Extracción de pares del Columbia dataset
├── train_gaze_model.py         # Entrenamiento U-Net (para Google Colab)
├── cuantizador.py              # Cuantización INT8 de modelos
├── extractor.py                # Extracción de frames de calibración
├── test_ipc.py                 # Test manual del canal IPC
├── CMakeLists.txt              # Build raíz
└── README.md
```

---

## Modelos ONNX

| Modelo | Archivo | Input | Peso | Latencia | Función |
|---|---|---|---|---|---|
| UltraFace | `version-slim-320_simplified.onnx` | [1,3,240,320] | 1.05 MB | 3.5-5 ms | Detección facial |
| PFLD | `pfld.onnx` | [1,3,112,112] | 2.98 MB | 17-19 ms | 68 landmarks faciales |
| MediaPipe Iris | `iris_landmark.onnx` | [1,3,64,64] | 2.65 MB | 5-6 ms | Centro del iris (diagnóstico) |
| GazeCorrector | `gaze_corrector.onnx` | [1,3,64,64] | 4.27 MB | ~3 ms/ojo | Corrección de mirada |

> **Nota:** Los modelos no se incluyen en el repositorio (`.gitignore`). Deben descargarse o generarse localmente.

---

## Protocolo IPC

Protocolo binario de 1 byte entre daemon y cliente:

| Byte | Dirección | Significado |
|---|---|---|
| `0x00` | Cliente → Daemon | Desactivar corrección |
| `0x01` | Cliente → Daemon | Activar corrección |
| `0x03` | Daemon → Cliente | Alarma térmica (Nivel 3) |

---

## Entrenamiento del Modelo de Corrección

El modelo `gaze_corrector.onnx` se entrena con el [Columbia Gaze Dataset](http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) (56 sujetos, 21 direcciones de mirada).

### Pasos

1. **Descargar** el Columbia Gaze Dataset
2. **Extraer pares:** `python extract_training_pairs.py` (genera ~11,200 pares en `dataset_ojos/`)
3. **Subir** `dataset_ojos/` a Google Drive
4. **Entrenar** en Google Colab con GPU T4: copiar `train_gaze_model.py` en una celda y ejecutar (~20 min)
5. **Descargar** `gaze_corrector.onnx` a `modelos/`

### Arquitectura del modelo

- **Tipo:** U-Net residual compacta
- **Parámetros:** ~1.1M
- **Principio:** `output = clamp(input + correction, 0, 1)`
- **Loss:** L1 (pixel-wise) + Perceptual (VGG16 features)
- **Augmentation:** flip, rotación ±5°, jitter brillo/contraste, ruido gaussiano, blur aleatorio

---

## Estado del Desarrollo

| Sprint | Descripción | Estado |
|---|---|---|
| 0 | Infraestructura Zero-Copy | ✅ Cerrado |
| 1 | Inferencia ONNX Runtime | ✅ Cerrado |
| 2 | Servicio Continuo + IPC | ✅ Cerrado |
| 3 | Warping Geométrico | ⚠️ Reemplazado por GazeNN |
| 4 | Supervivencia Térmica | ✅ Cerrado |
| 5 | Cliente GUI Multiplataforma | ✅ Cerrado |
| 6 | CI/CD + Empaquetado | 🔲 Pendiente |
| **7** | **Entrenamiento GAN** | **🔜 Próximo** |

### Sprint 7: Entrenamiento GAN (Próximo)

La corrección actual es insuficiente porque L1 loss produce correcciones conservadoras. El siguiente paso es reentrenar con un discriminador adversario (PatchGAN) que penalice outputs sin corrección visible. Cambio estimado: ~50 líneas adicionales en el script de Colab, 20 minutos de entrenamiento. Mismos datos.

### Optimizaciones futuras

- Cuantización INT8 de PFLD (17ms → ~5ms)
- Cuantización INT8 de GazeCorrector (4.27 MB → ~1.2 MB)
- Corrección del offset de solvePnP (~60° en pitch)
- Datasets adicionales: ETH-XGaze, GazeCapture

---

## Licencia

Propietario. Todos los derechos reservados.
