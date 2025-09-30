from ultralytics import YOLO
import cv2

# Carga modelo pre-entrenado
model = YOLO("yolov8n.pt")

cam_index = 3  # NVIDIA Broadcast u otra cámara
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

# Fuerza formato y tamaño “seguros” (opcional)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Warm-up
for _ in range(10):
    cap.read()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Sin frame de la cámara.")
        break

    # inferencia
    results = model.predict(frame, device=0, conf=0.25, verbose=False)

    # dibuja cajas y overlays
    annotated = results[0].plot()

    # Busca "person" y agrega coordenadas
    for b in results[0].boxes:
        cls_id = int(b.cls)
        cls_name = results[0].names[cls_id]

        if cls_name == "person":
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords_text = f"person: ({x1},{y1},{x2},{y2})"
            # esquina superior izquierda
            cv2.putText(
                annotated,
                coords_text,
                (10, 30),  # posición (x,y) en la ventana
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,          # tamaño
                (0, 255, 0),  # color (BGR)
                2,            # grosor
                cv2.LINE_AA
            )
            break  # solo la primera persona encontrada

    cv2.imshow("YOLO + NVIDIA Broadcast (q para salir)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
