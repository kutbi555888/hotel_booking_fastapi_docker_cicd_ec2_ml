# Hotel Booking Cancellation — End-to-End ML + FastAPI + Docker + CI/CD + AWS EC2

## Loyiha maqsadi

Bu loyiha `hotel_bookings_updated_2024.csv` dataset asosida booking cancel bo‘ladimi yoki yo‘qmi degan classification muammosini yechadi. Loyiha faqat model train qilish bilan tugamaydi. Bu yerda to‘liq oqim bor:

- datani ko‘rib chiqish
- EDA
- data preprocessing
- baseline model
- improvement model
- final evaluation
- best model save
- modular programming
- FastAPI
- Docker
- CI/CD
- AWS EC2

Bu versiyada **tuning yo‘q**. `merged` papkasi ham ishlatilmaydi.

---

## Dataset bo‘yicha qisqa xulosa

Dataset tekshirilganda quyidagilar aniqlandi:

- qatorlar soni: **119,390**
- ustunlar soni: **33**
- target ustuni: **`is_canceled`**
- cancel bo‘lmagan bookinglar: **62.96%**
- cancel bo‘lgan bookinglar: **37.04%**
- `company` missing: **94.31%**
- `agent` missing: **13.69%**
- `country` missing: **0.41%**
- `children` missing: juda kam
- `arrival_date_year` constant ustun
- `reservation_status` va `reservation_status_date` leakage xavfiga ega

Shu sababli modelingda leakage bo‘lishi mumkin bo‘lgan ustunlar chiqarib tashlangan va yangi feature lar qurilgan.

---

## Modeling bosqichlari

### 1) EDA
`scripts/01_eda.py`

Nima qiladi:
- missing summary
- numeric summary
- categorical summary
- target distribution
- oylik cancellation rate
- grafiklar
- `reports/eda/eda_summary.md`

### 2) Data preprocessing
`scripts/02_data_preprocessing.py`

Nima qiladi:
- train / validation / test bo‘ladi
- split stratified holatda qilinadi
- leakage ustunlari modelingga berilmaydi

### 3) Baseline model
`scripts/03_baseline_train.py`

Ichida:
- `DummyClassifier`
- `LogisticRegression`

### 4) Improvement model
`scripts/04_improvement_train.py`

Ichida:
- `XGBoost`

### 5) Final evaluation
`scripts/05_final_evaluation.py`

Nima qiladi:
- final model taqqoslanadi
- metriclar saqlanadi
- confusion matrix, ROC, PR curve yaratiladi

### 6) Best model save
`scripts/06_save_best_model.py`

Natija:
- `models/best_model.joblib`

---

## Yakuniy natija

Final test set natijalari:

### `improvement_xgboost`
- Accuracy: **0.8700**
- Precision: **0.8500**
- Recall: **0.7882**
- F1-score: **0.8179**
- ROC-AUC: **0.9455**

### `baseline_logreg`
- Accuracy: **0.8217**
- Precision: **0.7290**
- Recall: **0.8254**
- F1-score: **0.7742**
- ROC-AUC: **0.9093**

Tanlangan final model:
- **`improvement_xgboost`**
- saqlangan fayl: **`models/best_model.joblib`**

---

# Siz so‘ragan tartib aynan mana shu:
# 1. Requirements.txt (pipreqs)
# 2. FastAPI
# 3. Docker
# 4. CI/CD
# 5. AWS (EC2)

---

## 1. Requirements.txt (pipreqs)

Root ichida `requirements.txt` bor.

Uni oddiy install qilish:
```bash
pip install -r requirements.txt
```

Agar aynan `pipreqs` orqali qayta generatsiya qilmoqchi bo‘lsangiz:
```bash
chmod +x scripts/generate_requirements_pipreqs.sh
./scripts/generate_requirements_pipreqs.sh
```

Yoki qo‘lda:
```bash
pip install pipreqs
pipreqs . --force --ignore .venv,data,models,reports,.pytest_cache,__pycache__
```

---

## 2. FastAPI

FastAPI fayli:
- `app/main.py`

Endpointlar:
- `GET /`
- `GET /health`
- `GET /example`
- `POST /predict`

### FastAPI ni lokal ishga tushirish

#### 1-qadam. Virtual environment yaratish
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

#### 2-qadam. Dependency larni o‘rnatish
```bash
pip install -r requirements.txt
```

#### 3-qadam. ML pipeline ni ishga tushirish
Bu qadam modelni qayta tayyorlaydi:
```bash
python scripts/run_all.py
```

#### 4-qadam. FastAPI ni ishga tushirish
```bash
uvicorn app.main:app --reload
```

#### 5-qadam. Brauzerda ochish
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### Prediction request namunasi
```json
{
  "booking": {
    "hotel": "City Hotel - Seoul",
    "lead_time": 45,
    "arrival_date_month": "July",
    "arrival_date_week_number": 29,
    "arrival_date_day_of_month": 18,
    "stays_in_weekend_nights": 2,
    "stays_in_week_nights": 3,
    "adults": 2,
    "children": 1,
    "babies": 0,
    "meal": "BB",
    "country": "KOR",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 1,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": 9,
    "company": null,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 110.5,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
    "city": "Seoul"
  },
  "threshold": 0.5
}
```

`curl` bilan:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## 3. Docker

Docker fayllari:
- `Dockerfile`
- `docker-entrypoint.sh`
- `.dockerignore`

Bu versiyada Docker container ishga tushganda `models/best_model.joblib` bo‘lmasa, avtomatik `python scripts/run_all.py` ishga tushadi. Demak container modelni tayyorlab, keyin API ni ochadi.

### Docker build
```bash
docker build -t hotel-booking-api:latest .
```

### Docker run
```bash
docker run --name hotel-booking-api -p 8000:8000 hotel-booking-api:latest
```

Agar fonda ishlatmoqchi bo‘lsangiz:
```bash
docker run -d --name hotel-booking-api -p 8000:8000 hotel-booking-api:latest
```

Container tekshirish:
```bash
docker ps
```

Log ko‘rish:
```bash
docker logs -f hotel-booking-api
```

To‘xtatish:
```bash
docker stop hotel-booking-api
docker rm hotel-booking-api
```

---

## 4. CI/CD

Workflow fayli:
- `.github/workflows/ci-cd.yml`

Bu workflow quyidagilarni qiladi:

### CI qismi
- repository ni oladi
- Python 3.11 o‘rnatadi
- `requirements.txt` bo‘yicha dependency larni o‘rnatadi
- `compileall` bilan syntax tekshiradi
- `pytest -q` ni ishlatadi

### Docker build qismi
- Docker image build qiladi

### CD qismi
- faqat `main` branch ga push bo‘lganda ishlaydi
- agar quyidagi GitHub Secrets berilgan bo‘lsa, EC2 ga SSH orqali ulanadi:
  - `EC2_HOST`
  - `EC2_USER`
  - `EC2_SSH_KEY`

- keyin EC2 ichida:
  - `git pull`
  - `deployment/ec2_update_container.sh`

### GitHub Actions ishlashi uchun kerak bo‘ladigan secrets
Repository `Settings -> Secrets and variables -> Actions` ichida:
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

---

## 5. AWS (EC2)

Bu loyiha uchun **AWS Fargate emas, EC2 tanlangan**.

Deployment fayllari:
- `deployment/ec2_install_docker.sh`
- `deployment/ec2_run_container.sh`
- `deployment/ec2_update_container.sh`
- `deployment/nginx_hotel_booking.conf`

### EC2 da birinchi marta qilish kerak bo‘lgan ishlar

#### 1-qadam. EC2 instance ochish
Tavsiya:
- OS: **Ubuntu 22.04 LTS**
- Instance type: **t3.small** yoki kamida **t3.micro**
- Storage: **16 GB+**
- key pair yarating va `.pem` faylni saqlang

#### 2-qadam. Security Group
Oching:
- `22` port — SSH
- `80` port — HTTP
- `8000` port — test uchun ixtiyoriy

#### 3-qadam. EC2 ga SSH bilan kirish
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

#### 4-qadam. Repository ni clone qilish
```bash
git clone YOUR_REPOSITORY_URL
cd hotel_booking_fastapi_docker_cicd_ec2_ml
```

#### 5-qadam. Docker va Nginx o‘rnatish
```bash
chmod +x deployment/ec2_install_docker.sh
./deployment/ec2_install_docker.sh
```

#### 6-qadam. Container ni birinchi marta ishga tushirish
```bash
chmod +x deployment/ec2_run_container.sh
./deployment/ec2_run_container.sh
```

#### 7-qadam. Brauzerda tekshirish
- `http://YOUR_EC2_PUBLIC_IP`
- `http://YOUR_EC2_PUBLIC_IP/docs`
- `http://YOUR_EC2_PUBLIC_IP/health`

### EC2 da yangilash tartibi
Kod o‘zgarsa:
```bash
cd ~/hotel_booking_fastapi_docker_cicd_ec2_ml
chmod +x deployment/ec2_update_container.sh
./deployment/ec2_update_container.sh
```

### Docker holatini tekshirish
```bash
sudo docker ps
sudo docker logs -f hotel-booking-api
```

### Nginx holatini tekshirish
```bash
sudo systemctl status nginx
```

---

## Modular programming nega ishlatilgan

Kod notebook ichida aralash yozilmagan. U modullarga bo‘lingan:

- `src/hotel_booking_ml/data/`
- `src/hotel_booking_ml/features/`
- `src/hotel_booking_ml/preprocessing/`
- `src/hotel_booking_ml/models/`
- `src/hotel_booking_ml/evaluation/`
- `src/hotel_booking_ml/inference/`
- `src/hotel_booking_ml/utils/`

Bu yondashuvning foydasi:
- kod o‘qilishi oson bo‘ladi
- test yozish oson bo‘ladi
- API bilan ulash oson bo‘ladi
- Docker ga joylash oson bo‘ladi
- EC2 ga deploy qilish oson bo‘ladi

---

## Loyiha tuzilmasi

```text
hotel_booking_fastapi_docker_cicd_ec2_ml/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── engineered/
│   └── final/
│
├── deployment/
│   ├── ec2_install_docker.sh
│   ├── ec2_run_container.sh
│   ├── ec2_update_container.sh
│   └── nginx_hotel_booking.conf
│
├── models/
├── reports/
├── scripts/
│   ├── 01_eda.py
│   ├── 02_data_preprocessing.py
│   ├── 03_baseline_train.py
│   ├── 04_improvement_train.py
│   ├── 05_final_evaluation.py
│   ├── 06_save_best_model.py
│   ├── run_all.py
│   └── generate_requirements_pipreqs.sh
│
├── src/
├── tests/
├── Dockerfile
├── docker-entrypoint.sh
├── requirements.txt
└── README.md
```

---

## Eng tez ishga tushirish tartibi

### Lokal
```bash
pip install -r requirements.txt
python scripts/run_all.py
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t hotel-booking-api:latest .
docker run -p 8000:8000 hotel-booking-api:latest
```

### EC2
```bash
chmod +x deployment/ec2_install_docker.sh
./deployment/ec2_install_docker.sh

chmod +x deployment/ec2_run_container.sh
./deployment/ec2_run_container.sh
```

---

## Muhim eslatmalar

1. `merged` papkasi kerak emas va ishlatilmaydi.  
2. `tuning` bu versiyada yo‘q.  
3. `best_model.joblib` prediction uchun ishlatiladi.  
4. Docker container model bo‘lmasa o‘zi pipeline ishlatadi.  
5. EC2 deployment Docker + Nginx oqimida qurilgan.  
