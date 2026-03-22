import os
import glob
import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict

def generate_priors(img_w, img_h):
    priors = []
    strides = [8, 16, 32, 64]
    min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
    for i in range(4):
        fm_h = int(np.ceil(img_h / strides[i]))
        fm_w = int(np.ceil(img_w / strides[i]))
        for y in range(fm_h):
            for x in range(fm_w):
                for mb in min_boxes[i]:
                    cx = (x + 0.5) * strides[i] / img_w
                    cy = (y + 0.5) * strides[i] / img_h
                    w = mb / img_w
                    h = mb / img_h
                    priors.append([cx, cy, w, h])
    return np.array(priors, dtype=np.float32)

def detect_face(frame, session, input_name, priors):
    h_frame, w_frame = frame.shape[:2]
    
    resized = cv2.resize(frame, (320, 240))
    img_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_data = (img_data.astype(np.float32) - 127.0) / 128.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    results = session.run(None, {input_name: img_data})
    scores, boxes = results[0][0], results[1][0]
    
    best_idx = np.argmax(scores[:, 1])
    if scores[best_idx, 1] < 0.7:
        return None
        
    pcx, pcy, pw, ph = priors[best_idx]
    bx, by, bw, bh = boxes[best_idx]
    cx = pcx + bx * 0.1 * pw
    cy = pcy + by * 0.1 * ph
    w = pw * np.exp(bw * 0.2)
    h = ph * np.exp(bh * 0.2)
    
    x1 = int((cx - w / 2) * w_frame)
    y1 = int((cy - h / 2) * h_frame)
    x2 = int((cx + w / 2) * w_frame)
    y2 = int((cy + h / 2) * h_frame)
    
    # Padding de +10% conservador para asegurar captacion de contornos PFLD
    pad_w, pad_h = (x2 - x1) // 10, (y2 - y1) // 10
    x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    x2, y2 = min(w_frame, x2 + pad_w), min(h_frame, y2 + pad_h)
    
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return None
        
    return (x1, y1, x2, y2), face_crop

def get_landmarks(face_crop, session, input_name):
    crop_resized = cv2.resize(face_crop, (112, 112))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_data = crop_rgb.astype(np.float32) / 255.0
    crop_data = np.transpose(crop_data, (2, 0, 1))
    crop_data = np.expand_dims(crop_data, axis=0)
    
    res = session.run(None, {input_name: crop_data})
    landmarks = res[0][0]
    return landmarks

def extract_eye(frame, bbox, landmarks, start_idx, end_idx):
    face_x1, face_y1, face_x2, face_y2 = bbox
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1
    h_f, w_f = frame.shape[:2]
    
    min_x, max_x = float('inf'), 0.0
    min_y, max_y = float('inf'), 0.0
    
    for i in range(start_idx, end_idx + 1):
        lx, ly = landmarks[i * 2], landmarks[i * 2 + 1]
        
        # Desnormalizacion: proyeccion absoluta al plano global del fotograma
        abs_x = face_x1 + int(lx * face_w)
        abs_y = face_y1 + int(ly * face_h)
        
        if abs_x < min_x: min_x = abs_x
        if abs_x > max_x: max_x = abs_x
        if abs_y < min_y: min_y = abs_y
        if abs_y > max_y: max_y = abs_y
        
    ew = max_x - min_x
    eh = max_y - min_y
    
    # +20% área central de padding volumétrico
    pad_x = int(ew * 0.2)
    pad_y = int(eh * 0.2)
    
    # Clamping restrictivo contra bordes globales
    abs_x1 = max(0, int(min_x) - pad_x)
    abs_y1 = max(0, int(min_y) - pad_y)
    abs_x2 = min(w_f, int(max_x) + pad_x)
    abs_y2 = min(h_f, int(max_y) + pad_y)
    
    if abs_x2 > abs_x1 and abs_y2 > abs_y1:
        crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            return cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    return None

def main():
    dataset_root = "/home/arch-adrian/columbia_gaze/Columbia Gaze Data Set"
    output_dir = "dataset_ojos"
    input_dir = os.path.join(output_dir, "input")
    target_dir = os.path.join(output_dir, "target")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    face_model_path = "/home/arch-adrian/directlook/modelos_fp32/version-slim-320_simplified.onnx"
    pfld_model_path = "/home/arch-adrian/directlook/modelos_fp32/pfld.onnx"
        
    face_session = ort.InferenceSession(face_model_path, providers=['CPUExecutionProvider'])
    pfld_session = ort.InferenceSession(pfld_model_path, providers=['CPUExecutionProvider'])
    
    face_input_name = face_session.get_inputs()[0].name
    pfld_input_name = pfld_session.get_inputs()[0].name
    
    priors = generate_priors(320, 240)
    
    all_images = glob.glob(os.path.join(dataset_root, "**", "*.jpg"), recursive=True)
    groups = defaultdict(list)
    
    # Análisis Léxico de Estructura Columbia Gaze
    for img_path in all_images:
        filename = os.path.basename(img_path)
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) >= 5:
            img_id = parts[0]
            pose = parts[2]
            gaze_v = parts[3]
            gaze_h = parts[4]
            key = (img_id, pose)
            groups[key].append({
                "path": img_path,
                "is_target": (gaze_v == "0V" and gaze_h == "0H")
            })
            
    global_pair_idx = 1
    
    for key, images in groups.items():
        # Filtro estricto: Descartar silbidos sin Ground Truth Target
        target_img_info = next((img for img in images if img["is_target"]), None)
        if target_img_info is None:
            continue
            
        target_path = target_img_info["path"]
        input_images = [img for img in images if not img["is_target"]]
        
        for input_img_info in input_images:
            input_path = input_img_info["path"]
            
            frame_in = cv2.imread(input_path)
            frame_tgt = cv2.imread(target_path)
            
            if frame_in is None or frame_tgt is None:
                continue
                
            in_face = detect_face(frame_in, face_session, face_input_name, priors)
            tgt_face = detect_face(frame_tgt, face_session, face_input_name, priors)
            
            if in_face is None or tgt_face is None:
                continue
                
            bbox_in, crop_in = in_face
            bbox_tgt, crop_tgt = tgt_face
            
            lm_in = get_landmarks(crop_in, pfld_session, pfld_input_name)
            lm_tgt = get_landmarks(crop_tgt, pfld_session, pfld_input_name)
            
            # Sub-extracción ocular mediante iteración asimétrica PFLD 68-puntos (36-41 // 42-47)
            l_eye_in = extract_eye(frame_in, bbox_in, lm_in, 36, 41)
            r_eye_in = extract_eye(frame_in, bbox_in, lm_in, 42, 47)

            l_eye_tgt = extract_eye(frame_tgt, bbox_tgt, lm_tgt, 36, 41)
            r_eye_tgt = extract_eye(frame_tgt, bbox_tgt, lm_tgt, 42, 47)
            
            # Exclusión geométrica atómica y sincronización de Nomenclatura Estricta 1:1
            if l_eye_in is not None and l_eye_tgt is not None:
                cv2.imwrite(os.path.join(input_dir, f'pair_{global_pair_idx:05d}_L.jpg'), l_eye_in)
                cv2.imwrite(os.path.join(target_dir, f'pair_{global_pair_idx:05d}_L.jpg'), l_eye_tgt)
                global_pair_idx += 1
                if global_pair_idx % 100 == 0:
                    print(f"[EXTRACCIÓN] Pares generados y guardados: {global_pair_idx}")
                
            if r_eye_in is not None and r_eye_tgt is not None:
                cv2.imwrite(os.path.join(input_dir, f'pair_{global_pair_idx:05d}_R.jpg'), r_eye_in)
                cv2.imwrite(os.path.join(target_dir, f'pair_{global_pair_idx:05d}_R.jpg'), r_eye_tgt)
                global_pair_idx += 1
                if global_pair_idx % 100 == 0:
                    print(f"[EXTRACCIÓN] Pares generados y guardados: {global_pair_idx}")

if __name__ == '__main__':
    main()
