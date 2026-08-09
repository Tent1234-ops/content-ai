# Content AI Analysis Recommendation System

เว็บแอปสำหรับวิเคราะห์คลิปของผู้ใช้ สกัด transcript/keywords จัดประเภทคอนเทนต์ เปรียบเทียบกับ dataset ของคลิป engagement สูง และแนะนำ keyword gap กับความยาววิดีโอที่เหมาะสม

## Tech Stack

- Backend: FastAPI, SQLAlchemy, JWT auth
- Database: MySQL ตาม `sql/schema.sql` หรือ SQLite fallback ผ่าน `app.db`
- Jobs: in-process worker เป็นค่าเริ่มต้น, รองรับ RQ/Redis แบบ optional
- Frontend: Flutter
- Analysis: Speech-to-Text pipeline, NLP keyword extraction, clustering, recommendation engine

## Backend Setup

1. สร้าง virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. ตั้งค่า environment

```powershell
Copy-Item .env.example .env
```

ค่าแนะนำสำหรับ demo แบบง่าย:

```env
DATABASE_URL=sqlite:///./app.db
JWT_SECRET=change-this-secret
ADMIN_INVITE_CODE=admin-demo
YOUTUBE_API_KEY=
YOUTUBE_REGION=TH
GOOGLE_REGION=TH
TIKTOK_REGION=TH
ASR_LOCAL_FILES_ONLY=1
ANALYZE_MAX_AUDIO_SECONDS=60
ANALYZE_ENABLE_THAI_SPELL_CORRECTION=0
ANALYZE_ENABLE_ML_KEYWORDS=0
ANALYZE_ENABLE_VISUAL=0
NLP_USE_PYTHAINLP=0
```

3. รัน backend

```powershell
uvicorn app.main:app --reload
```

Backend จะเปิดที่ `http://127.0.0.1:8000`

## Database Setup

### SQLite Demo

เพื่อให้ root โปรเจคสะอาด จะไม่เก็บ `app.db` ไว้ใน root repository แล้ว ให้ตั้ง
`DATABASE_URL=sqlite:///./app.db` แล้วรัน backend หนึ่งครั้งเพื่อให้ SQLAlchemy
สร้าง/ปรับ schema อัตโนมัติ จากนั้นรัน seed script เพื่อเติมข้อมูล demo

```powershell
uvicorn app.main:app --reload
python scripts/seed_demo_dataset.py
```

ถ้าต้องอ้างอิงฐานข้อมูล demo เดิม ไฟล์ถูกย้ายไปที่ `archive/runtime_artifacts/app.db`

### MySQL

1. สร้าง schema จากไฟล์:

```powershell
mysql -u root -p < sql/schema.sql
```

2. ตั้งค่า `.env`:

```env
DB_DRIVER=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=content_ai
DB_USER=root
DB_PASSWORD=1234
```

## Seed Demo Dataset

เติม dataset ตัวอย่างสำหรับ recommendation:

```powershell
python scripts/seed_demo_dataset.py
```

Dataset จะมีหมวดตัวอย่าง เช่น `smartphone`, `food_drink`, `skincare`, `audio`

## Optional RQ Worker

ค่าเริ่มต้นใช้ in-process worker ไม่ต้องเปิด worker แยก ถ้าต้องการใช้ RQ:

```powershell
pip install rq redis
python scripts/start_rq_worker.py --redis redis://localhost:6379/0 --queue default
```

จากนั้นใน Admin Settings ตั้ง `job_backend` เป็น `rq` และ `redis_url` ให้ตรงกัน

## Frontend Setup

```powershell
cd frontend_flutter
flutter pub get
flutter run -d chrome --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

`API_BASE_URL` defaults to `http://127.0.0.1:8000`. Set it with
`--dart-define` when the backend runs on another host or port.

## Faster Whisper Setup

Prepare the multilingual `small` model once before starting the demo backend:

```powershell
python scripts/setup_faster_whisper.py --model small
python scripts/setup_faster_whisper.py --model small --verify-only
```

Production/demo configuration uses the verified local model without downloading
during an upload:

```env
ASR_MODEL_SIZE=small
ASR_MODEL_DIR=models_cache/faster_whisper
ASR_LOCAL_FILES_ONLY=1
ASR_REQUIRE_MODEL_READY=1
ASR_LANGUAGE=auto
ANALYZE_MAX_AUDIO_SECONDS=60
ANALYZE_ENABLE_THAI_SPELL_CORRECTION=0
```

`ASR_LANGUAGE=auto` lets Faster Whisper detect Thai or English. If speech-to-text
fails, the pipeline uses the filename only as a low-confidence fallback and shows
a warning in Recommendation Evidence.

Thai dictionary spell correction is disabled by default because applying it to
long ASR phrases is CPU- and memory-intensive. Domain term normalization still
runs. Enable it only for short, low-confidence transcripts that need an
additional correction pass.

Seed 24 evidence clips for each supported category (168 rows total):

```powershell
python scripts/seed_demo_dataset.py
```

## Phase 11 Notification Migration

Run this once when upgrading an existing database:

```powershell
python scripts/migrate_phase11_notifications.py
```

The migration archives the previous notification tables and creates
`user_trend_watch_sessions` plus the session-scoped `notifications` table.
Login establishes the current trend snapshot as a baseline without sending an
alert. Later dashboard polls create one `new_live_trend` notification for each
trend key first seen during that login session. Uploaded-video analysis does not
create notifications.

ค่า API backend อยู่ใน `frontend_flutter/lib/services/api_client.dart` ค่าเริ่มต้นคือ:

```dart
http://127.0.0.1:8000
```

## Demo Flow

1. Register/Login
   - เปิด Flutter app
   - สมัคร user ปกติผ่านหน้า Register
   - Login เพื่อเข้า Dashboard

2. สร้าง Admin
   - สมัครด้วย invite code ที่ตั้งใน `.env` เช่น `ADMIN_INVITE_CODE=admin-demo`
   - Login ด้วย admin account
   - เข้า `Admin Console`

3. Admin Sync Dataset
   - จาก Dashboard กดปุ่ม `Sync live trends` หรือเข้า endpoint sync trends
   - ระบบดึง YouTube/Google/TikTok แบบ live และบันทึกลง `dataset_contents`
   - Admin ดู/เพิ่ม/แก้ dataset ได้ที่ `Admin Datasets`

4. User Upload Video
   - Login เป็น user
   - เข้า `Analyze My Clip`
   - อัปโหลดวิดีโอหรือไฟล์เสียง
   - Backend สร้าง `job_id`

5. Job Processing
   - Frontend polling `/jobs/{job_id}`
   - แสดงสถานะ `queued`, `running`, `completed`, `failed`
   - เมื่อ completed จะ parse result ไปหน้า Result

6. Result Recommendation
   - หน้า Result แสดงข้อมูลสำคัญ:
     - ประเภทคลิป
     - keywords ที่ผู้ใช้มี
     - keyword gap จาก dataset engagement สูงในหมวดเดียวกัน
     - recommended duration ของหมวดนั้น
     - evidence/source/sample size เพื่ออธิบายที่มาคำแนะนำ

7. Save / History / Dashboard
   - กด Save to My Ideas เพื่อบันทึกผล
   - ดูย้อนหลังใน `History / My Ideas`
   - Dashboard แสดง live trends จาก YouTube/Google/TikTok
   - Follow topic เพื่อรับ notification เมื่อ trend ใหม่ตรงกับหัวข้อที่ติดตาม

## Admin Scope Checklist

- ดู/เพิ่ม/อัปเดต dataset: `Admin Datasets`
- ตั้งค่า max keywords, hook duration, scan interval: `Admin Console`
- ดู logs สำเร็จ/ล้มเหลว: `System Logs`
- ดู cluster runs และ run clustering: `Cluster Runs`
- ดู report เปรียบเทียบ YouTube/TikTok/Google: `Admin Console`

## Useful Endpoints

- `POST /auth/register`
- `POST /auth/login`
- `POST /analyze/save`
- `GET /jobs/{job_id}`
- `POST /trends/youtube/sync`
- `POST /trends/google/sync`
- `POST /trends/tiktok/sync`
- `GET /dashboard/summary?trend_mode=live`
- `GET /admin/datasets`
- `POST /admin/datasets`
- `PUT /admin/datasets/{dataset_id}`
- `GET /admin/logs`
- `GET /admin/clusters/runs`
- `GET /recommendations/admin/report`
