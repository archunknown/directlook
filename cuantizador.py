import os
import cv2
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader

class CalibradorDirectLook(CalibrationDataReader):
    def __init__(self, calibration_image_folder, input_name, is_pfld):
        self.image_folder = calibration_image_folder
        self.input_name = input_name
        self.is_pfld = is_pfld
        self.image_list = [os.path.join(calibration_image_folder, f) 
                           for f in os.listdir(calibration_image_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.enum_data = iter(self.image_list)

    def get_next(self):
        try:
            image_path = next(self.enum_data)
        except StopIteration:
            return None

        img = cv2.imread(image_path)
        if img is None:
            return self.get_next()

        if self.is_pfld:
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
        else:
            img = cv2.resize(img, (320, 240))
            img = (img.astype(np.float32) - 127.0) / 128.0

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return {self.input_name: img}

def get_input_name(model_path):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session.get_inputs()[0].name

os.makedirs('build/modelos', exist_ok=True)

# Proceso 1: PFLD
pfld_input = get_input_name('modelos_fp32/pfld.onnx')
pfld_dr = CalibradorDirectLook('datos_calibracion', pfld_input, is_pfld=True)
quantize_static(
    'modelos_fp32/pfld.onnx',
    'build/modelos/pfld.onnx',
    pfld_dr,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)

# Proceso 2: UltraFace (slim-320)
ultraface_input = get_input_name('modelos_fp32/version-slim-320_simplified.onnx')
ultraface_dr = CalibradorDirectLook('datos_calibracion', ultraface_input, is_pfld=False)
quantize_static(
    'modelos_fp32/version-slim-320_simplified.onnx',
    'build/modelos/version-slim-320_simplified.onnx',
    ultraface_dr,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8
)
