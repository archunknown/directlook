import cv2
import os

if not os.path.exists('rostro.mp4'):
    raise RuntimeError("El archivo rostro.mp4 no existe o no se puede abrir.")

cap = cv2.VideoCapture('rostro.mp4')
if not cap.isOpened():
    raise RuntimeError("El archivo rostro.mp4 no existe o no se puede abrir.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    raise RuntimeError("El video no tiene fotogramas o no se pudo leer correctamente.")

os.makedirs('datos_calibracion', exist_ok=True)

interval = total_frames / 100.0
extracted = 0

for i in range(100):
    frame_id = int(i * interval)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join('datos_calibracion', f'calib_{extracted:03d}.jpg'), frame)
        extracted += 1

cap.release()
print(f"Se extrajeron {extracted} fotogramas y se guardaron en 'datos_calibracion/'")
