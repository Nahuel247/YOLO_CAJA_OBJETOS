# YOLO Webcam Overlay — coordenadas de person en vivo

Muchas veces necesitamos realizar análisis sobre partes específicas de una imagen o video (p.ej., solo personas, matrículas, cajas, etc.).  
**YOLO** es una excelente opción para encontrar rápidamente esos objetos y luego encadenar un análisis posterior.  

Este repo muestra un ejemplo mínimo con **Ultralytics YOLOv8** + **OpenCV** que detecta personas en una cámara web y dibuja las **coordenadas (x1,y1,x2,y2)** de la primera detección en la esquina superior izquierda de la ventana.

👉 https://github.com/tu_usuario/yolo-webcam-overlay  
*(reemplaza con tu URL cuando lo subas)*

---

## ✨ Demo rápida

- Abre una ventana llamada:  
  **YOLO + NVIDIA Broadcast (q para salir)**
- Dibuja cajas sobre los objetos detectados.
- Si hay al menos una persona, muestra el texto en verde:  
  `person: (x1,y1,x2,y2)`
- Pulsa **q** para salir.

---

## 📦 Requisitos

- **Python 3.9–3.11**
- **NVIDIA GPU** (opcional pero recomendado). Probado con RTX 4070 Laptop + CUDA 12.1
- **Windows 11** (funciona también en Linux/macOS ajustando el backend de cámara)

### Librerías
- `ultralytics` (YOLOv8)
- `opencv-python`
- (Opcional GPU) `torch` con CUDA compatible

> 💡 Si usas GPU, instala PyTorch con el wheel de CUDA adecuado (ej.: `cu121`).  
> Verifica la instalación con:  
> ```bash
> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> ```
