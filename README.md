# YOLO Webcam Overlay — coordenadas de person en vivo

Muchas veces necesitamos realizar análisis sobre partes específicas de una imagen o video (p.ej., solo personas, matrículas, cajas, etc.).  
**YOLO** es una excelente opción para encontrar rápidamente esos objetos y luego encadenar un análisis posterior.  

Este repo muestra un ejemplo mínimo con **Ultralytics YOLOv8** + **OpenCV** que detecta personas en una cámara web y dibuja las **coordenadas (x1,y1,x2,y2)** de la primera detección en la esquina superior izquierda de la ventana.

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

```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121


import torch, platform
print("PyTorch:", torch.__version__, "| Compilado con CUDA:", torch.version.cuda, "| CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
