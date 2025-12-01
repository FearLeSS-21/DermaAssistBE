# 🌿 DermaAssist Backend

**AI-Powered Dermatology Detection API**  
Built with Django + MySQL + DigitalOcean

---

## 📋 Overview

DermaAssist Backend is a comprehensive dermatology analysis API that processes user-uploaded facial images to detect and analyze multiple skin conditions:

- **Acne** - Using YOLOv8
- **Wrinkles** - Using Roboflow models
- **Eyebags** - Using Roboflow models
- **Eczema** - Using Roboflow models

The system leverages advanced computer vision techniques including YOLO, Roboflow, MediaPipe, and OpenCV for accurate detection. All user data, analysis history, and progress tracking are stored in a MySQL database.

**Deployment:** DigitalOcean Droplets with Nginx + Gunicorn/Uvicorn

---

## ✨ Features

### 🔍 **Skin Condition Detection**
- **AI Models:**
  - YOLOv8 for acne detection
  - Roboflow hosted models for eyebags, wrinkles, and eczema
  - MediaPipe FaceMesh and Selfie Segmentation
- **Advanced Processing:**
  - Automatic face alignment and cropping
  - Region-based classification (forehead, nose, chin, cheeks, under eyes)

### 📊 **User Progress Tracking**
- Saves detection history to MySQL database
- Generates progress plots and charts for each user
- Displays improvement trends over time
- JSON-based backup system (`user_progress.json`)

### 📷 **Real-Time Camera Stream**
- Accepts base64-encoded video frames
- Performs real-time analysis on streaming data

### 🛍️ **Product Recommendations**
- AI-generated recommendations based on severity levels (mild/moderate)
- Integrated product search from scraped skincare database

### 🔍 **Product Search Engine**
- Search functionality for skincare products
- Data sourced from `eparkville_skincare_playwright.csv`
- Categorized results

### 🔐 **User Authentication**
- Simple authentication system with User ID and Password
- Signup, Login, and Logout endpoints

### 🗄️ **MySQL Database**
Stores comprehensive user data:
- User profiles
- Progress reports
- Image analysis records

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Django / Django REST Framework** | API backend framework |
| **MySQL** | Primary database |
| **ASGI + WSGI** | Async and sync request handling |
| **DigitalOcean Droplet** | Cloud hosting infrastructure |
| **MediaPipe** | Facial landmark detection & segmentation |
| **YOLOv8 (Ultralytics)** | Acne detection model |
| **Roboflow** | Hosted models for eczema, wrinkles, eyebags |
| **OpenCV** | Image processing operations |
| **Pillow** | Image manipulation |
| **Matplotlib** | Progress chart generation |
| **ThreadPoolExecutor** | Parallel model inference |

---

## 📂 Project Structure

```
/DermaAssistBackend
│
├── api/
│   ├── views.py              # Full AI processing pipeline
│   ├── models.py             # Database models (Image, UserProfile, Progress)
│   ├── serializers.py        # DRF serializers
│   └── ...
│
├── media/
│   └── Project_Folder/       # Uploaded and processed images
│
├── user_progress.json        # JSON logs for user progress
├── requirements.txt          # Python dependencies
├── manage.py                 # Django management script
├── .env                      # Environment variables (create this)
└── README.md                 # This file
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd DermaAssistBackend
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure MySQL Database

**Create the database:**

```sql
CREATE DATABASE dermaassist;
```

**Update `settings.py`:**

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'dermaassist',
        'USER': 'root',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

### 5️⃣ Run Database Migrations

```bash
python manage.py migrate
```

### 6️⃣ Create Environment Variables

Create a `.env` file in the project root:

```env
ROBOFLOW_EYEBAGS_API_KEY=your_api_key_here
ROBOFLOW_WRINKLE_API_KEY=your_api_key_here
ROBOFLOW_ECZEMA_API_KEY=your_api_key_here
PRODUCT_CSV_PATH=path/to/eparkville_skincare_playwright.csv
```

### 7️⃣ Start Development Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000`

---

## 🌐 Production Deployment (DigitalOcean)

### Prerequisites

SSH into your DigitalOcean Droplet and install required packages:

```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx mysql-server
```

### Setup Steps

**1. Create production environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Collect static files:**

```bash
python manage.py collectstatic
```

**3. Run with Gunicorn (WSGI):**

```bash
gunicorn --bind 0.0.0.0:8000 yourproject.wsgi
```

**OR run with Uvicorn (ASGI):**

```bash
uvicorn yourproject.asgi:application --host 0.0.0.0 --port 8000
```

**4. Configure Nginx:**

Create `/etc/nginx/sites-available/dermaassist`:

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/your/staticfiles/;
    }

    location /media/ {
        alias /path/to/your/media/;
    }
}
```

**5. Enable the site and restart Nginx:**

```bash
sudo ln -s /etc/nginx/sites-available/dermaassist /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔗 API Endpoints

### 📤 **Upload Image**
- **Endpoint:** `POST /upload-image/`
- **Description:** Upload a facial image for analysis
- **Returns:**
  - Processed image
  - Detected conditions
  - Cropped region URLs
  - Product recommendations
  - Progress plot

### 📷 **Camera Stream**
- **Endpoint:** `POST /camera-stream/`
- **Description:** Stream base64 frames for real-time analysis

### 📈 **Get Progress**
- **Endpoint:** `GET /progress/?user_id=<id>`
- **Description:** Retrieve user's historical progress data

### 🔍 **Product Search**
- **Endpoint:** `GET /product-search/?query=cleanser`
- **Description:** Search for skincare products

### 🔑 **User Signup**
- **Endpoint:** `POST /signup/`
- **Description:** Create a new user account

### 🔑 **User Login**
- **Endpoint:** `POST /login/`
- **Description:** Authenticate existing user

### 🚪 **User Logout**
- **Endpoint:** `POST /logout/`
- **Description:** End user session

### 📝 **All Reports**
- **Endpoint:** `GET /reports/`
- **Description:** Retrieve all analysis reports

---

## 🎯 Future Improvements

- [ ] Implement JWT authentication for enhanced security
- [ ] Dockerize the application for easier deployment
- [ ] Add background task processing with Celery + Redis
- [ ] Optimize mobile streaming performance
- [ ] Expand detection to include:
  - Rosacea
  - Hyperpigmentation
  - Dark spots
  - Other skin conditions

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

[Add your license information here]

---

## ❤️ Acknowledgments

This project was built using the following technologies and services:

- **Django** - Web framework
- **MySQL** - Database management
- **DigitalOcean** - Cloud hosting
- **YOLO (Ultralytics)** - Object detection
- **Roboflow** - Computer vision models
- **MediaPipe** - Face mesh and segmentation
- **Android Front-End** - Built with Jetpack Compose

---

## 📧 Contact

[Add your contact information or links here]

---

**Made with ❤️ for better skin health**