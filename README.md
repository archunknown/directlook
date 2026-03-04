# DirectLook (Arquitectura de Dominio Dividido)

> Herramienta nativa para Windows y Linux que simula contacto visual directo en tiempo real mediante manipulación de la mirada (warping).

---

## Estructura de Aislamiento

La arquitectura se divide estrictamente en **dos componentes aislados** para proteger el rendimiento del procesamiento de video. El aislamiento de procesos es innegociable.

| Componente | Descripción | Restricciones |
|---|---|---|
| **Motor (Daemon Core)** | Servicio nativo C++ sin interfaz gráfica | Latencias `<30ms` · RAM `<80MB` |
| **Controlador (Cliente GUI)** | Interfaz ultraligera en la bandeja del sistema (System Tray) para el usuario final | Prohibido tocar el buffer de video |

Ambos componentes se comunican **exclusivamente vía IPC**. La interfaz gráfica tiene prohibido tocar el buffer de video.

---

## Prerrequisitos de Hardware y Software

- **Procesador:** Soporte para instrucciones `AVX2`
- **Lenguaje base:** C++17/20
- **Sistema de construcción:** CMake

### Entorno Linux

| Categoría | Tecnología |
|---|---|
| **Compilador** | GCC o Clang |
| **Visión y Matemáticas** | OpenCV (C++ API) |
| **Inferencia** | ONNX Runtime (C++ API) — modelos INT8 |
| **GUI Cliente** | Framework GTK3 minimalista |
| **Canal IPC** | Sockets de dominio UNIX |
| **Virtualización de Video** | Módulo del kernel `v4l2loopback` |

### Entorno Windows

| Categoría | Tecnología |
|---|---|
| **Compilador** | MSVC |
| **Visión y Matemáticas** | OpenCV (C++ API) |
| **Inferencia** | ONNX Runtime (C++ API) — modelos INT8 |
| **GUI Cliente** | Win32 API pura |
| **Canal IPC** | Named Pipes |
| **Virtualización de Video** | Filtro DirectShow o API de cámara virtual |

---

## Compilación (CMake)

El pipeline utiliza CMake para unificar los procesos de construcción en Windows y Linux.

**Pasos:**

1. Clonar el repositorio.
2. Descargar las dependencias de ONNX Runtime e introducirlas en el directorio correspondiente.
3. Posicionarse en la raíz del proyecto y generar el árbol de construcción:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
