import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from django.conf import settings

logger = logging.getLogger(__name__)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_KEY_HERE")

CLIENTS = {
    'eyebags': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY),
    'wrinkles': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY),
    'eczema': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY),
}

MODELS_CONFIG = {
    'eyebags': 'dark-circle-wj25f/1',
    'wrinkles': 'wrinkle-detection/2',
    'eczema': 'eczemadetection/1'
}

ACNE_MODEL = None
def get_acne_model():
    global ACNE_MODEL
    if ACNE_MODEL is None:
        try:
            ACNE_MODEL = YOLO("acne.pt")
        except Exception as e:
            logger.error(f"Could not load YOLO model: {e}")
    return ACNE_MODEL

def get_facial_region(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    if image is None:
        return None, None
        
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return image, None
            
        return image, results.multi_face_landmarks[0]

def analyze_roboflow(client_key, image_path, confidence=0.2):
    try:
        client = CLIENTS.get(client_key)
        model_id = MODELS_CONFIG.get(client_key)
        if not client or not model_id:
            return []
            
        res = client.infer(image_path, model_id=model_id)
        return [p for p in res.get('predictions', []) if p['confidence'] >= confidence]
    except Exception as e:
        logger.error(f"Roboflow error for {client_key}: {e}")
        return []

from ..models import Product
from ..serializers import ProductSerializer

def recommend_products(acne_severity, wrinkle_severity, eyebag_severity=0):
    recommendations = []
    
    if acne_severity > 0:
        acne_products = Product.objects.filter(skin_concern__iexact="Acne")[:2]
        recommendations.extend(acne_products)
        
    if wrinkle_severity > 0:
        wrinkle_products = Product.objects.filter(skin_concern__iexact="Wrinkles")[:2]
        recommendations.extend(wrinkle_products)

    if eyebag_severity > 0:
         eyebag_products = Product.objects.filter(skin_concern__iexact="Eyebags")[:2]
         recommendations.extend(eyebag_products)
    
    if not recommendations:
        general_products = Product.objects.filter(skin_concern__iexact="Other")[:3]
        recommendations.extend(general_products)
        
    return ProductSerializer(recommendations, many=True).data