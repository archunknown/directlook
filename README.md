# DirectLook (Arquitectura de Dominio Dividido)

[cite_start]Herramienta nativa para Windows y Linux que simula contacto visual directo en tiempo real mediante manipulación de la mirada (warping)[cite: 4]. 

## Estructura de Aislamiento
[cite_start]La arquitectura se divide estrictamente en dos componentes aislados para proteger el rendimiento del procesamiento de video[cite: 6]. [cite_start]El aislamiento de procesos es innegociable[cite: 13].

1. [cite_start]**Motor (Daemon Core):** Servicio nativo C++ sin interfaz gráfica[cite: 7]. [cite_start]Restricciones estrictas: latencias <30ms y consumo de RAM <80MB[cite: 7].
2. [cite_start]**Controlador (Cliente GUI):** Interfaz ultraligera en la bandeja del sistema (System Tray) para el usuario final[cite: 8].



[cite_start]Ambos componentes se comunican exclusivamente vía IPC[cite: 8]. [cite_start]La interfaz gráfica tiene prohibido tocar el buffer de video[cite: 13].

## Prerrequisitos de Hardware y Software
[cite_start]La compilación exige un procesador con soporte para instrucciones AVX2[cite: 19]. 
[cite_start]Lenguaje base: C++17/20[cite: 17]. 
[cite_start]Sistema de construcción: CMake[cite: 59].

### Entorno Linux
* [cite_start]**Compilador:** GCC o Clang[cite: 59].
* [cite_start]**Visión y Matemáticas:** OpenCV (C++ API)[cite: 18].
* [cite_start]**Inferencia:** ONNX Runtime (C++ API) para cargar modelos INT8[cite: 19].
* [cite_start]**GUI Cliente:** Framework GTK3 minimalista[cite: 21].
* [cite_start]**Canal IPC:** Sockets de dominio UNIX[cite: 23].
* [cite_start]**Virtualización de Video:** Módulo del kernel `v4l2loopback`[cite: 25].

### Entorno Windows
* [cite_start]**Compilador:** MSVC[cite: 59].
* [cite_start]**Visión y Matemáticas:** OpenCV (C++ API)[cite: 18].
* [cite_start]**Inferencia:** ONNX Runtime (C++ API) para cargar modelos INT8[cite: 19].
* [cite_start]**GUI Cliente:** Win32 API pura[cite: 21].
* [cite_start]**Canal IPC:** Named Pipes[cite: 23].
* [cite_start]**Virtualización de Video:** Filtro DirectShow o API de cámara virtual[cite: 26].

## Compilación (CMake)
[cite_start]El pipeline utiliza CMake para unificar los procesos de construcción en Windows y Linux[cite: 59]. 

1. Clonar el repositorio.
2. Descargar las dependencias de ONNX Runtime e introducirlas en el directorio correspondiente.
3. Posicionarse en la raíz del proyecto y generar el árbol de construcción:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release