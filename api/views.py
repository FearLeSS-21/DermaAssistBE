import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image as PILImage
import io
import json
from inference_sdk import InferenceHTTPClient
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import FileSystemStorage
from django.utils import timezone
from django.core.cache import cache
from ultralytics import YOLO
from .models import Image, UserProfile, Progress
from .serializers import ImageSerializer, UserProfileSerializer
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

CLIENTS = {
    'eyebags': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_EYEBAGS_API_KEY", "1Sv1nJWnSYTOgUFPI6ml")),
    'wrinkles': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_WRINKLE_API_KEY", "xfJxuiY45eNBjGUEcuom")),
    'eczema': InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=os.getenv("ROBOFLOW_ECZEMA_API_KEY", "xfJxuiY45eNBjGUEcuom"))
}

MODELS = {
    'eyebags': ('dark-circle-wj25f/1', 0.01, (0, 0, 255), "Possible under-eye puffiness; consider hydrating cream"),
    'wrinkles': ('wrinkle-detection/2', 0.001, (255, 0, 0), "Possible wrinkle; consider anti-aging cream"),
    'eczema': ('eczemadetection/1', 0.05, (0, 255, 255), "Possible eczema patch; consider soothing cream")
}

ACNE_MODEL = YOLO("acne.pt")
ACNE_CONFIDENCE = 0.20
ACNE_COLOR = (0, 255, 0)
ACNE_MESSAGE = "Possible acne spot; consider gentle cleansing"

BASE_OUTPUT_FOLDER = "media/Project_Folder"
FOLDERS = {
    'original': os.path.join(BASE_OUTPUT_FOLDER, "original"),
    'processed': os.path.join(BASE_OUTPUT_FOLDER, "processed"),
    'acne': os.path.join(BASE_OUTPUT_FOLDER, "processed", "acne"),
    'eyebags': os.path.join(BASE_OUTPUT_FOLDER, "processed", "eyebags"),
    'wrinkles': os.path.join(BASE_OUTPUT_FOLDER, "processed", "wrinkles"),
    'eczema': os.path.join(BASE_OUTPUT_FOLDER, "processed", "eczema"),
    'cropped_faces': os.path.join(BASE_OUTPUT_FOLDER, "cropped_faces"),
    'progress_plots': os.path.join(BASE_OUTPUT_FOLDER, "processed", "progress_plots")
}
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

PROGRESS_FILE = "user_progress.json"

PRODUCT_RECOMMENDATIONS = {
    "acne": [
        {"severity": "mild", "products": ["Starville Acne Prone Skin Facial Cleanser 200 ml (EGP 105.00)", "Starville Acne-Prone Skin Hydrate Gel 100ML"]},
        {"severity": "moderate", "products": ["Starville Acne Prone Skin Facial Cleanser 400 ml (EGP 180.00)", "Starville Acne-Prone Skin Hydrate Gel 100ML"]},
    ],
    "wrinkles": [
        {"severity": "mild", "products": ["GLAMY LAB AntiWrinkles Gel 50 gm (EGP 400.00)"]},
        {"severity": "moderate", "products": ["GLAMY LAB AntiWrinkles Gel 50 gm (EGP 400.00)"]},
    ],
    "eyebags": [
        {"severity": "mild", "products": ["Starville Acne-Prone Skin Hydrate Gel 100ML"]},
        {"severity": "moderate", "products": ["Starville Acne-Prone Skin Hydrate Gel 100ML"]},
    ],
    "eczema": [
        {"severity": "mild", "products": ["Starville Soothing Cream for Sensitive Skin 100ML (EGP 150.00)"]},
        {"severity": "moderate", "products": ["Starville Intensive Moisturizing Cream 200ML (EGP 250.00)"]},
    ]
}

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
logger = logging.getLogger(__name__)

def get_facial_region(x, y, landmarks, face_width, face_height, scale_factor=1.0):
    points = {
        "Forehead": landmarks[10], "Nose": landmarks[2], "Chin": landmarks[152],
        "Left Eye": landmarks[33], "Right Eye": landmarks[263],
        "Left Cheek": landmarks[234], "Right Cheek": landmarks[454]
    }
    pixels = {k: (int(v.x * face_width), int(v.y * face_height)) for k, v in points.items()}
    regions = [
        ("Forehead", pixels["Forehead"][0] - face_width * 0.3 * scale_factor, 
         pixels["Forehead"][0] + face_width * 0.3 * scale_factor,
         pixels["Forehead"][1] - face_height * 0.1 * scale_factor, 
         pixels["Forehead"][1] + face_height * 0.2 * scale_factor),
        ("Nose", pixels["Nose"][0] - face_width * 0.1 * scale_factor, 
         pixels["Nose"][0] + face_width * 0.1 * scale_factor,
         pixels["Forehead"][1] + face_height * 0.2 * scale_factor, 
         pixels["Nose"][1] + face_height * 0.1 * scale_factor),
        ("Chin", pixels["Chin"][0] - face_width * 0.3 * scale_factor, 
         pixels["Chin"][0] + face_width * 0.3 * scale_factor,
         pixels["Nose"][1] + face_height * 0.2 * scale_factor, 
         pixels["Chin"][1] + face_height * 0.2 * scale_factor),
        ("Left Cheek", pixels["Left Cheek"][0] - face_width * 0.2 * scale_factor, 
         pixels["Left Cheek"][0] + face_width * 0.2 * scale_factor,
         pixels["Left Eye"][1], pixels["Left Cheek"][1] + face_height * 0.2 * scale_factor),
        ("Right Cheek", pixels["Right Cheek"][0] - face_width * 0.2 * scale_factor, 
         pixels["Right Cheek"][0] + face_width * 0.2 * scale_factor,
         pixels["Right Eye"][1], pixels["Right Cheek"][1] + face_height * 0.2 * scale_factor),
        ("Under Left Eye", pixels["Left Eye"][0] - face_width * 0.15 * scale_factor, 
         pixels["Left Eye"][0] + face_width * 0.15 * scale_factor,
         pixels["Left Eye"][1], pixels["Left Eye"][1] + face_height * 0.1 * scale_factor),
        ("Under Right Eye", pixels["Right Eye"][0] - face_width * 0.15 * scale_factor, 
         pixels["Right Eye"][0] + face_width * 0.15 * scale_factor,
         pixels["Right Eye"][1], pixels["Right Eye"][1] + face_height * 0.1 * scale_factor)
    ]
    for name, x_min, x_max, y_min, y_max in regions:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return name
    region_centers = [((x_min + x_max) / 2, (y_min + y_max) / 2) for _, x_min, x_max, y_min, y_max in regions]
    distances = [(i, ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) for i, (cx, cy) in enumerate(region_centers)]
    return regions[min(distances, key=lambda x: x[1])[0]][0] if distances else "Unknown"

def determine_severity(confidence, num_detections):
    return "moderate" if confidence >= 0.8 or num_detections > 5 else "mild"

def recommend_products(detections, condition_type):
    cache_key = f"recommend_{condition_type}{len(detections)}{max((d['confidence'] for d in detections), default=0)}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    products = next((rec["products"] for rec in PRODUCT_RECOMMENDATIONS.get(condition_type, [])
                     if rec["severity"] == determine_severity(max((d['confidence'] for d in detections), default=0), len(detections))), [])
    cache.set(cache_key, products, timeout=3600)
    return products

def save_zoomed_crop(image, x, y, w, h, folder, filename, fs, zoom_factor=1.0):
    x1, y1 = max(0, int(x - w * zoom_factor)), max(0, int(y - h * zoom_factor))
    x2, y2 = min(image.shape[1], int(x + w * zoom_factor)), min(image.shape[0], int(y + h * zoom_factor))
    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid crop dimensions")
        return None
    crop = image[y1:y2, x1:x2]
    if not crop.size:
        logger.warning("Empty crop for %s", filename)
        return None
    crop_path = fs.path(os.path.join(folder, filename))
    cv2.imwrite(crop_path, crop)
    return crop_path

def validate_image(image_file):
    try:
        img = PILImage.open(io.BytesIO(image_file.read()))
        img.verify()
        image_file.seek(0)
        return True
    except Exception as e:
        logger.error("Invalid image: %s", str(e))
        return False

def load_json_progress(user_id):
    try:
        if not os.path.exists(PROGRESS_FILE):
            return []
        with open(PROGRESS_FILE, 'r') as f:
            return [json.loads(line.strip()) for line in f if line.strip() and json.loads(line.strip()).get('user_id') == user_id]
    except Exception as e:
        logger.error("Failed to load progress JSON: %s", e)
        return []

def generate_progress_plot(user_id, fs, timestamp):
    try:
        json_entries = load_json_progress(user_id)
        if not json_entries:
            return None
        timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts = [], [], [], [], []
        for entry in json_entries:
            try:
                ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                detections = entry.get('detections', {})
                timestamps.append(ts)
                acne_counts.append(len(detections.get('acne', [])))
                wrinkles_counts.append(len(detections.get('wrinkles', [])))
                eyebags_counts.append(len(detections.get('eyebags', [])))
                eczema_counts.append(len(detections.get('eczema', [])))
            except (ValueError, KeyError):
                continue
        if not timestamps:
            return None
        sorted_data = sorted(zip(timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts), key=lambda x: x[0])
        timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts = zip(*sorted_data)
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, acne_counts, label='Acne', marker='o', color='green')
        plt.plot(timestamps, wrinkles_counts, label='Wrinkles', marker='o', color='red')
        plt.plot(timestamps, eyebags_counts, label='Eyebags', marker='o', color='blue')
        plt.plot(timestamps, eczema_counts, label='Eczema', marker='o', color='cyan')
        plt.xlabel('Timestamp')
        plt.ylabel('Number of Detections')
        plt.title(f'Progress histology for User {user_id}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        os.makedirs(FOLDERS['progress_plots'], exist_ok=True)
        plot_filename = f"processed/progress_plots/progress_{user_id}_{timestamp}.png"
        plot_absolute_path = os.path.join(BASE_OUTPUT_FOLDER, plot_filename)
        with open(plot_absolute_path, 'wb') as f:
            f.write(buffer.getvalue())
        return fs.url(os.path.join("Project_Folder", plot_filename))
    except Exception as e:
        logger.error("Progress plot generation failed: %s", e)
        return None

class UploadImageView(APIView):
    def post(self, request):
        if not request.META.get('CONTENT_TYPE', '').startswith('multipart/form-data'):
            return Response({'error': "Request must use 'multipart/form-data' content type"}, status=status.HTTP_400_BAD_REQUEST)
        image_file = request.FILES.get('file')
        if not image_file:
            return Response({'error': "No image provided"}, status=status.HTTP_400_BAD_REQUEST)
        if not validate_image(image_file) or image_file.size > 10 * 1024 * 1024:
            return Response({'error': 'Invalid image or size exceeds 10MB'}, status=status.HTTP_400_BAD_REQUEST)
        name = request.data.get('name', 'Unnamed')
        user_id = request.data.get('user_id', 'default_user')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fs = FileSystemStorage(location='media')
        ext = image_file.content_type.split('/')[-1].lower()
        if ext not in ['jpeg', 'jpg', 'png', 'bmp', 'gif']:
            ext = 'jpg'
        try:
            original_path = fs.save(f"Project_Folder/original/image_{timestamp}.{ext}", image_file)
            image_path = fs.path(original_path)
            pil_image = PILImage.open(image_path).convert('RGB')
            # Ensure correct orientation (hair facing up)
            exif = pil_image.getexif()
            orientation = exif.get(274, 1)  # 274 is the EXIF tag for orientation
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            h, w = image.shape[:2]
            if h < 100 or w < 100 or np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) < 10:
                return Response({'error': 'Image too small or low contrast'}, status=status.HTTP_400_BAD_REQUEST)
            with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = (selfie.process(image_rgb).segmentation_mask > 0.1).astype(np.uint8) * 255
                image = cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(np.ones_like(image) * 255, np.ones_like(image) * 255, mask=cv2.bitwise_not(mask))
            with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3, static_image_mode=True) as face_mesh:
                results = face_mesh.process(image_rgb)
                if not results.multi_face_landmarks:
                    return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)
                landmarks = results.multi_face_landmarks[0].landmark
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                x_center, y_center = (min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2
                face_width = max(x_coords) - min(x_coords)
                face_height = max(y_coords) - min(y_coords)
                # Estimate distance based on face size relative to image
                face_size_ratio = max(face_width, face_height) / max(w, h)
                if face_size_ratio > 0.5:  # Too close
                    zoom_factor = 1.0
                    distance_status = "too_close"
                elif face_size_ratio < 0.2:  # Too far
                    zoom_factor = 2.0
                    distance_status = "too_far"
                else:  # Normal distance
                    zoom_factor = 1.5
                    distance_status = "normal"
                padding = max(face_width, face_height) * 0.2
                x_min, x_max = max(0, int(x_center - face_width / 2 - padding)), min(w, int(x_center + face_width / 2 + padding))
                y_min, y_max = max(0, int(y_center - face_height / 2 - padding)), min(h, int(y_center + face_height / 2 + padding))
                face_image = image[y_min:y_max, x_min:x_max]
                if face_image.size == 0 or face_image.shape[0] < 50 or face_image.shape[1] < 50:
                    return Response({'error': 'Invalid face crop'}, status=status.HTTP_400_BAD_REQUEST)
            _, face_buffer = cv2.imencode(f'.{ext}', face_image)
            face_path = fs.save(f"Project_Folder/cropped_faces/face_{timestamp}.{ext}", io.BytesIO(face_buffer.tobytes()))
            face_url = fs.url(f"Project_Folder/cropped_faces/face_{timestamp}.{ext}")
            face_size = max(face_image.shape[0], face_image.shape[1])
            target_size = (720, 720) if face_size > 500 else (480, 480)
            face_image_resized = cv2.resize(face_image, target_size)
            scale_x, scale_y = face_image.shape[1] / target_size[0], face_image.shape[0] / target_size[1]
            temp_path = fs.path(f"Project_Folder/processed/temp_{timestamp}.{ext}")
            cv2.imwrite(temp_path, face_image_resized)
            annotated_face = face_image.copy()
            detections = {"acne": [], "wrinkles": [], "eyebags": [], "eczema": []}
            def process_detections(condition, response, min_conf, folder, hint, color):
                if not isinstance(response, dict) or "predictions" not in response:
                    return []
                raw_detections = [{
                    "x": int(pred["x"] * scale_x), "y": int(pred["y"] * scale_y),
                    "width": int(pred["width"] * scale_x * zoom_factor),
                    "height": int(pred["height"] * scale_y * zoom_factor),
                    "confidence": float(pred["confidence"]), "class": pred.get("class", "unknown")
                } for pred in response["predictions"]]
                filtered_detections = [det for det in raw_detections if det["confidence"] >= min_conf and (condition != "eczema" or det["class"].lower() == "eczema")]
                result = []
                for idx, det in enumerate(filtered_detections, 1):
                    x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                    region = get_facial_region(x, y, landmarks, face_image.shape[1], face_image.shape[0], zoom_factor)
                    crop_path = save_zoomed_crop(face_image, x, y, w, h, folder, f"{condition}{timestamp}{idx}.{ext}", fs, zoom_factor)
                    result.append({
                        "id": idx, "x": x, "y": y, "width": w, "height": h,
                        "confidence": det["confidence"], "region": region, "hint": hint,
                        "crop_url": fs.url(f"Project_Folder/processed/{condition}/{condition}{timestamp}{idx}.{ext}") if crop_path else None
                    })
                    cv2.rectangle(annotated_face, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                    cv2.putText(annotated_face, f"{condition.capitalize()} {idx}: {region} ({det['confidence']:.2f})",
                                (max(0, x - w // 2), max(20, y - h // 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                return result
            try:
                acne_results = ACNE_MODEL.predict(source=face_image_resized, conf=ACNE_CONFIDENCE)
                for idx, box in enumerate(acne_results[0].boxes, 1):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x, y = int((x1 + x2) / 2 * scale_x), int((y1 + y2) / 2 * scale_y)
                    w, h = int((x2 - x1) * scale_x * zoom_factor), int((y2 - y1) * scale_y * zoom_factor)
                    region = get_facial_region(x, y, landmarks, face_image.shape[1], face_image.shape[0], zoom_factor)
                    crop_path = save_zoomed_crop(face_image, x, y, w, h, FOLDERS['acne'], f"acne_{timestamp}_{idx}.{ext}", fs, zoom_factor)
                    detections['acne'].append({
                        "id": idx, "x": x, "y": y, "width": w, "height": h,
                        "confidence": float(box.conf[0]), "region": region, "hint": ACNE_MESSAGE,
                        "crop_url": fs.url(f"Project_Folder/processed/acne/acne_{timestamp}_{idx}.{ext}") if crop_path else None
                    })
                    cv2.rectangle(annotated_face, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), ACNE_COLOR, 2)
                    cv2.putText(annotated_face, f"Acne {idx}: {region} ({box.conf[0]:.2f})",
                                (max(0, x - w // 2), max(20, y - h // 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACNE_COLOR, 2)
            except Exception as e:
                logger.error("Acne detection failed: %s", e)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {condition: executor.submit(process_detections, condition, CLIENTS[condition].infer(temp_path, model_id=model_id), min_conf, FOLDERS[condition], hint, color)
                           for condition, (model_id, min_conf, color, hint) in MODELS.items()}
                for condition, future in futures.items():
                    try:
                        detections[condition] = future.result()
                    except Exception as e:
                        logger.error("%s detection failed: %s", condition.capitalize(), e)
            progress = Progress.objects.create(
                user_id=user_id,
                acne=[{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in d.items()} for d in detections["acne"]],
                wrinkles=[{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in d.items()} for d in detections["wrinkles"]],
                eyebags=[{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in d.items()} for d in detections["eyebags"]],
                eczema=[{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in d.items()} for d in detections["eczema"]]
            )
            progress_plot_url = generate_progress_plot(user_id, fs, timestamp)
            if progress_plot_url:
                progress.progress_plot = f"Project_Folder/processed/progress_plots/progress_{user_id}_{timestamp}.png"
                progress.save()
            with open(PROGRESS_FILE, 'a') as f:
                json.dump({
                    'timestamp': timezone.now().isoformat(),
                    'user_id': user_id,
                    'detections': detections,
                    'image_path': f"processed_{timestamp}.{ext}",
                    'progress_plot_url': progress_plot_url,
                    'distance_status': distance_status
                }, f)
                f.write('\n')
            _, processed_buffer = cv2.imencode(f'.{ext}', annotated_face)
            processed_path = fs.save(f"Project_Folder/processed/processed_{timestamp}.{ext}", io.BytesIO(processed_buffer.tobytes()))
            processed_url = fs.url(f"Project_Folder/processed/processed_{timestamp}.{ext}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            image_record = Image.objects.create(
                image=original_path, processed_image=processed_path,
                name=name, request_type='POST', ip_address=request.META.get('REMOTE_ADDR')
            )
            detection_status = {cond: {"count": len(dets), "details": dets, "message": f"{len(dets)} detected" if dets else "Zero detected"}
                               for cond, dets in detections.items()}
            response_data = {
                "message": "Image processed",
                "data": ImageSerializer(image_record, context={'request': request}).data,
                "original_image_url": fs.url(original_path),
                "processed_image_url": processed_url,
                "face_image_url": face_url,
                "detections": detection_status,
                "recommendations": {cond: recommend_products(dets, cond) for cond, dets in detections.items()},
                "progress_plot_url": progress_plot_url,
                "distance_status": distance_status
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error("Processing error: %s", e)
            return Response({'error': f'Processing failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CameraStreamView(APIView):
    def post(self, request):
        try:
            frame_data = request.data.get('frame')
            if not frame_data:
                return Response({'error': 'No frame data provided'}, status=status.HTTP_400_BAD_REQUEST)
            nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return Response({'error': 'Invalid frame data'}, status=status.HTTP_400_BAD_REQUEST)
            # Ensure correct orientation (hair facing up) using face landmarks
            with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3, static_image_mode=True) as face_mesh:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    forehead_y = landmarks[10].y  # Forehead landmark
                    chin_y = landmarks[152].y  # Chin landmark
                    if forehead_y > chin_y:  # If forehead is below chin, rotate 180 degrees
                        image = cv2.rotate(image, cv2.ROTATE_180)
            image_file = io.BytesIO()
            _, buffer = cv2.imencode('.jpg', image)
            image_file.write(buffer.tobytes())
            image_file.name = 'camera_frame.jpg'
            image_file.seek(0)
            request.FILES['file'] = image_file
            return UploadImageView().post(request)
        except Exception as e:
            logger.error("Camera stream error: %s", e)
            return Response({'error': f'Camera stream error: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProgressView(APIView):
    def get(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'User ID required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            entries = Progress.objects.filter(user_id=user_id).order_by('timestamp')
            if not entries:
                return Response({'message': 'No progress data'}, status=status.HTTP_200_OK)
            progress_summary = [{
                "timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "acne": {"count": len(entry.acne or []), "avg_confidence": sum(d["confidence"] for d in entry.acne) / len(entry.acne) if entry.acne else 0},
                "wrinkles": {"count": len(entry.wrinkles or []), "avg_confidence": sum(d["confidence"] for d in entry.wrinkles) / len(entry.wrinkles) if entry.wrinkles else 0},
                "eyebags": {"count": len(entry.eyebags or []), "avg_confidence": sum(d["confidence"] for d in entry.eyebags) / len(entry.eyebags) if entry.eyebags else 0},
                "eczema": {"count": len(entry.eczema or []), "avg_confidence": sum(d["confidence"] for d in entry.eczema) / len(entry.eczema) if entry.eczema else 0}
            } for entry in entries]
            improvement = {"acne": 0, "wrinkles": 0, "eyebags": 0, "eczema": 0}
            if len(entries) >= 2:
                latest, previous = entries[-1], entries[-2]
                improvement = {
                    "acne": len(previous.acne or []) - len(latest.acne or []),
                    "wrinkles": len(previous.wrinkles or []) - len(latest.wrinkles or []),
                    "eyebags": len(previous.eyebags or []) - len(latest.eyebags or []),
                    "eczema": len(previous.eczema or []) - len(latest.eczema or [])
                }
            fs = FileSystemStorage(location='media')
            progress_plot_url = generate_progress_plot(user_id, fs, datetime.now().strftime("%Y%m%d_%H%M%S"))
            response_data = {'progress': progress_summary, 'improvement': improvement}
            if progress_plot_url:
                response_data['progress_plot_url'] = progress_plot_url
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Progress error: %s", e)
            return Response({'error': f'Progress error: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserProfileView(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        password = request.data.get('password')
        if not user_id or not password:
            return Response({'error': 'User ID and password required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = UserProfile.objects.get(user_id=user_id)
            if user.password == password:
                request.session['user_id'] = user_id
                return Response({'message': 'Login successful', 'data': UserProfileSerializer(user).data}, status=status.HTTP_200_OK)
            return Response({'error': 'Incorrect password'}, status=status.HTTP_401_UNAUTHORIZED)
        except UserProfile.DoesNotExist:
            serializer = UserProfileSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                request.session['user_id'] = user_id
                return Response({'message': 'Profile created', 'data': serializer.data}, status=status.HTTP_201_CREATED)
            return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

class SignupView(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        password = request.data.get('password')
        if not user_id or not password:
            return Response({'error': 'User ID and password required'}, status=status.HTTP_400_BAD_REQUEST)
        if UserProfile.objects.filter(user_id=user_id).exists():
            return Response({'error': 'User already exists'}, status=status.HTTP_400_BAD_REQUEST)
        serializer = UserProfileSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            request.session['user_id'] = user_id
            return Response({'message': 'Profile created', 'data': serializer.data}, status=status.HTTP_201_CREATED)
        return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        user_id = request.data.get('user_id')
        password = request.data.get('password')
        if not user_id or not password:
            return Response({'error': 'User ID and password required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = UserProfile.objects.get(user_id=user_id)
            if user.password == password:
                request.session['user_id'] = user_id
                if password == 'default_password':
                    return Response({'message': 'Login successful, please change default password', 'data': UserProfileSerializer(user).data}, status=status.HTTP_200_OK)
                return Response({'message': 'Login successful', 'data': UserProfileSerializer(user).data}, status=status.HTTP_200_OK)
            return Response({'error': 'Incorrect password'}, status=status.HTTP_401_UNAUTHORIZED)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User does not exist'}, status=status.HTTP_404_NOT_FOUND)

class LogoutView(APIView):
    def post(self, request):
        if request.session.get('user_id'):
            del request.session['user_id']
        return Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)

class AllReports(APIView):
    def get(self, request):
        try:
            reports = [{
                'user_id': e['user_id'],
                'date': e['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'detections': {cond: len(e[cond] or []) for cond in ['acne', 'wrinkles', 'eyebags', 'eczema']}
            } for e in Progress.objects.all().values('timestamp', 'acne', 'wrinkles', 'eyebags', 'eczema', 'user_id')]
            if not reports:
                return Response({'error': 'No reports found'}, status=status.HTTP_404_NOT_FOUND)
            return Response(reports, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Error fetching reports: %s", str(e))
            return Response({'error': f'Failed to retrieve reports: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProductSearchView(APIView):
    def get(self, request):
        query = request.GET.get('query', '').strip().lower()
        try:
            csv_path = os.getenv('PRODUCT_CSV_PATH', '/home/GradProject2025NU/GProject/webscrappinng/eparkville_skincare_playwright.csv')
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
            df['target'] = df['target'].str.lower().fillna('other').replace({
                'acne-related': 'acne', 'anti-aging': 'wrinkles', 'anti aging': 'wrinkles',
                'dark circle': 'eyebags', 'dark circles': 'eyebags', 'under-eye': 'eyebags',
                'dermatitis': 'eczema', 'misc': 'other', 'general': 'other'
            })
            if query:
                df = df[df['name'].str.lower().str.contains(query, na=False) | df['target'].str.lower().str.contains(query, na=False)]
            products_by_category = {
                k: df[df['target'].str.contains(v, na=False)][['name', 'price', 'url']].to_dict('records')
                for k, v in [('Acne', 'acne'), ('Wrinkles', 'wrinkle|anti-aging'), ('Eyebags', 'eyebag|dark circle|under-eye'),
                             ('Eczema', 'eczema|dermatitis'), ('Other', 'other|general|misc')]
            }
            products_by_category = {k: v for k, v in products_by_category.items() if v}
            if not products_by_category:
                return Response({'query': query, 'products_by_category': {}, 'message': f'No products found for "{query or "No query"}"'}, status=status.HTTP_404_NOT_FOUND)
            return Response({'query': query, 'products_by_category': products_by_category, 'message': None}, status=status.HTTP_200_OK)
        except FileNotFoundError:
            return Response({'query': query, 'products_by_category': {}, 'message': 'No products available'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error("Search error: %s", str(e))
            return Response({'query': query, 'products_by_category': {}, 'message': 'Search error'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProgressCompare(APIView):
    def get(self, request):
        try:
            entries = Progress.objects.all().order_by('timestamp')
            if not entries:
                return Response({'message': 'No progress data available'}, status=status.HTTP_200_OK)
            timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts = [], [], [], [], []
            for entry in entries:
                try:
                    timestamps.append(entry.timestamp)
                    acne_counts.append(len(entry.acne or []))
                    wrinkles_counts.append(len(entry.wrinkles or []))
                    eyebags_counts.append(len(entry.eyebags or []))
                    eczema_counts.append(len(entry.eczema or []))
                except Exception:
                    continue
            if not timestamps:
                return Response({'message': 'No valid progress data for comparison'}, status=status.HTTP_200_OK)
            sorted_data = sorted(zip(timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts), key=lambda x: x[0])
            timestamps, acne_counts, wrinkles_counts, eyebags_counts, eczema_counts = zip(*sorted_data)
            fs = FileSystemStorage(location='media')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.figure(figsize=(12, 8))
            plt.plot(timestamps, acne_counts, label='Acne', marker='o', color='green')
            plt.plot(timestamps, wrinkles_counts, label='Wrinkles', marker='o', color='red')
            plt.plot(timestamps, eyebags_counts, label='Eyebags', marker='o', color='blue')
            plt.plot(timestamps, eczema_counts, label='Eczema', marker='o', color='cyan')
            plt.xlabel('Timestamp')
            plt.ylabel('Number of Detections')
            plt.title('Comparison of Skin Condition Detections')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            os.makedirs(FOLDERS['progress_plots'], exist_ok=True)
            plot_filename = f"processed/progress_plots/compare_all_{timestamp}.png"
            plot_absolute_path = os.path.join(BASE_OUTPUT_FOLDER, plot_filename)
            with open(plot_absolute_path, 'wb') as f:
                f.write(buffer.getvalue())
            plot_url = fs.url(os.path.join("Project_Folder", plot_filename))
            response_data = {
                'message': 'Comparison plot generated',
                'plot_url': plot_url,
                'summary': {
                    'total_entries': len(entries),
                    'average_counts': {
                        'acne': sum(acne_counts) / len(acne_counts) if acne_counts else 0,
                        'wrinkles': sum(wrinkles_counts) / len(wrinkles_counts) if wrinkles_counts else 0,
                        'eyebags': sum(eyebags_counts) / len(eyebags_counts) if eyebags_counts else 0,
                        'eczema': sum(eczema_counts) / len(eczema_counts) if eczema_counts else 0
                    }
                }
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Progress comparison error: %s", e)
            return Response({'error': f'Progress comparison error: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class Results(APIView):
    def get(self, request):
        try:
            # Retrieve only the latest entry based on timestamp
            entry = Progress.objects.latest('timestamp')
            # Define condition mapping
            conditions = [
                ('0', 'acne', entry.acne),
                ('1', 'eczema', entry.eczema),
                ('2', 'wrinkles', entry.wrinkles),
                ('3', 'eyebags', entry.eyebags)
            ]
            # Select the first condition with non-empty detections, or default to acne if none
            selected_condition = None
            for code, name, data in conditions:
                if data and len(data) > 0:
                    selected_condition = (code, name, data)
                    break
            if not selected_condition:
                selected_condition = ('0', 'acne', entry.acne or [])  # Default to acne if no detections
            
            code, name, data = selected_condition
            result = {
                'user_id': entry.user_id,
                'timestamp': entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'condition': {
                    'code': code,
                    'name': name,
                    'count': len(data),
                    'details': [{k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in d.items()} for d in data]
                }
            }
            response_data = {
                'message': 'Latest result retrieved successfully',
                'result': result
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Progress.DoesNotExist:
            return Response({'message': 'No results data available'}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Results retrieval error: %s", e)
            return Response({'error': f'Results retrieval error: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)