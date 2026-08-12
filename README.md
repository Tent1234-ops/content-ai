# Content AI Analysis Recommendation System

เว็บแอปสำหรับวิเคราะห์คลิปรีวิวความยาวไม่เกิน 5 นาที ถอดเสียงด้วย Faster Whisper
จำแนกหมวดหมู่แบบหลายระดับ สกัดคำสำคัญ และเปรียบเทียบกับคลิป YouTube
Creative Commons ในหมวดเดียวกันที่ผ่านการตรวจโดยมนุษย์ เพื่อแนะนำ keyword gap
และความยาวคลิปที่เหมาะสม

## System Components

- Backend: FastAPI, SQLAlchemy, background jobs
- Database: MySQL หรือ SQLite สำหรับ local development
- Frontend: Flutter Web
- Speech-to-Text: Faster Whisper multilingual `small`
- Live dashboard: YouTube, Google Trends และ TikTok แบบ partial result
- Recommendation dataset: YouTube Creative Commons + public transcript + human-reviewed label

## Important Dataset Policy

ระบบไม่มี demo seed สำหรับ Classification หรือ Recommendation อีกต่อไป ข้อมูลหนึ่งแถวจะถูกใช้จริงได้เมื่อผ่านเงื่อนไขทั้งหมด:

- YouTube API ยืนยัน `status.license=creativeCommon`
- วิดีโอยาวไม่เกิน 300 วินาที
- มี transcript ภาษาไทยหรืออังกฤษจาก public caption
- คนตรวจวิดีโอ, transcript และหมวดหมู่ แล้วระบุ `decision=approve`
- มี reviewer, reviewed time, transcript quality และ license provenance ครบ
- อยู่ใน taxonomy `content-taxonomy-v1`
- แบ่ง train/validation/test ตาม Channel ID เพื่อป้องกันข้อมูลช่องเดียวกันรั่วข้ามชุด

Search query ใช้ค้นหา candidate เท่านั้น ไม่ถือเป็น label จริง ระบบจะไม่เปิดหมวดให้ AI
จนกว่าจะมีข้อมูลที่ผ่าน human review อย่างน้อย 30 คลิปในหมวดนั้น

## Taxonomy V1

หมวด Level 3 รุ่นแรกมี 12 หมวด:

`Phone`, `Camera`, `Laptop`, `Audio`, `Headphone`, `Hardware`, `Food`,
`Drink`, `Makeup`, `Grooming`, `Shirt`, `Shoes`

คลิปที่ไม่อยู่ในขอบเขตหรือมีหลักฐานไม่พอจะได้ผลลัพธ์ `Unknown/Other`

## Backend Setup

```powershell
cd Z:\content-ai.worktrees\agents-content-analysis-recommendation-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

ค่าหลักใน `.env`:

```env
DB_DRIVER=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=content_ai
DB_USER=root
DB_PASSWORD=1234
JWT_SECRET=change-this-secret
ADMIN_INVITE_CODE=admin-local
YOUTUBE_API_KEY=your-youtube-data-api-key
YOUTUBE_REGION=TH
ASR_MODEL_SIZE=small
ASR_MODEL_DIR=models_cache/faster_whisper
ASR_LOCAL_FILES_ONLY=1
ASR_REQUIRE_MODEL_READY=1
ASR_LANGUAGE=auto
ANALYZE_MAX_AUDIO_SECONDS=300
```

เตรียมโมเดลและ schema:

```powershell
python -B scripts/setup_faster_whisper.py --model small
python -B scripts/setup_faster_whisper.py --model small --verify-only
python -B scripts/migrate_youtube_cc_dataset.py
```

รัน backend:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

ตรวจระบบที่ `http://127.0.0.1:8000/health`

สร้างบัญชี Admin ครั้งแรก คำสั่งจะถามรหัสผ่านโดยไม่แสดงหรือบันทึกลงไฟล์:

```powershell
python -B scripts/create_admin_user.py --username admin --email admin@example.com
```

## Flutter Setup

เปิด PowerShell อีกหน้าต่าง:

```powershell
cd Z:\content-ai.worktrees\agents-content-analysis-recommendation-system\frontend_flutter
flutter pub get
flutter run -d edge --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

## Build YouTube CC Dataset

ขั้นตอนนี้ไม่ดาวน์โหลดไฟล์วิดีโอ Collector ใช้ YouTube Data API เพื่อค้นหาและยืนยัน
Creative Commons metadata จากนั้นใช้ `youtube-transcript-api` เพื่ออ่าน public captions
ที่หน้า YouTube เปิดให้ดูได้

### 1. Collect Candidates

เริ่มทดสอบหนึ่งหมวดก่อน:

```powershell
python -B scripts/collect_youtube_cc_candidates.py `
  --leaves phone `
  --target-per-leaf 5 `
  --languages th,en `
  --region TH
```

เมื่อทดสอบผ่านแล้วจึงเก็บครบทุกหมวด:

```powershell
python -B scripts/collect_youtube_cc_candidates.py `
  --target-per-leaf 50 `
  --languages th,en `
  --region TH `
  --max-pages-per-query 2
```

ผลลัพธ์ถูกแยกเป็น:

- `data/raw/youtube_cc/<version>/candidates-*.jsonl`: metadata และ transcript จริง
- `data/reviews/youtube_cc/<version>/review-*.csv`: แบบฟอร์ม human review
- `data/manifests/youtube-cc-*.json`: hash, config และผลการเก็บข้อมูล
- `dataset_collection_runs`: audit ของแต่ละรอบ

### 2. Human Review

เข้าสู่ระบบด้วยบัญชี Admin แล้วเปิดเมนู `Dataset Review` หรือ route
`/#/admin-dataset-review` หน้าเว็บจะแสดงวิดีโอ, transcript, automated checks,
หมวดที่ระบบเสนอ และ coverage ปัจจุบัน กด `Approve` หรือ `Reject` ได้โดยตรง

เมื่อ Approve ต้องยืนยัน `reviewed_leaf_key` และ `transcript_quality`; ระบบใส่ชื่อ Admin
กับเวลาตรวจจาก login session ให้อัตโนมัติ แล้วบันทึกทั้ง `dataset_contents`,
`dataset_review_events`, `system_logs` และ review artifact สำหรับ audit

ไฟล์ `review-*.csv` ยังใช้เป็นช่องทางสำรองสำหรับการตรวจแบบออฟไลน์ โดยกรอกคอลัมน์:

- `decision`: `approve` หรือ `reject`
- `reviewed_leaf_key`: หนึ่งใน 12 leaf keys เมื่อ approve
- `transcript_quality`: `good` หรือ `acceptable` เมื่อ approve
- `reviewer`: ชื่อหรืออีเมลผู้ตรวจ
- `reviewed_at`: ISO 8601 เช่น `2026-08-12T10:30:00+07:00`
- `review_notes`: เหตุผลหรือสิ่งผิดปกติที่พบ

ห้ามกรอก approve อัตโนมัติจาก search query เพราะ search query เป็นเพียง candidate label

### 3. Import Reviewed Rows

ไม่ต้องรันคำสั่งนี้เมื่อกดปุ่มผ่านหน้า Dataset Review เพราะหน้าเว็บ import แถวที่ตัดสินใจ
ให้ทันที ใช้คำสั่งนี้เฉพาะ workflow CSV แบบออฟไลน์:

```powershell
python -B scripts/import_youtube_cc_reviews.py `
  --candidates "data\raw\youtube_cc\youtube-cc-th-v1\candidates-YYYYMMDD-HHMMSS.jsonl" `
  --reviews "data\reviews\youtube_cc\youtube-cc-th-v1\review-YYYYMMDD-HHMMSS.csv"
```

Importer ทำงานแบบ atomic หาก review ที่กรอกแล้วมีแม้แต่หนึ่งแถวไม่ถูกต้อง จะไม่บันทึก
แถวใดในรอบนั้นเลย รายการ approve จะลง `dataset_contents`; approve/reject ทั้งหมดจะมี audit
ใน `dataset_review_events`

### 4. Verify Readiness

```powershell
python -B scripts/verify_submission_dataset.py
python -B scripts/verify_submission_dataset.py --require-ready
```

ดูรายหมวดผ่าน `GET /classification/taxonomy` ค่า `ready=true` หมายถึงหมวดนั้นมีอย่างน้อย
30 ตัวอย่างจริงที่ผ่านกติกา production ไม่ได้หมายถึงโมเดลผ่านเกณฑ์ความแม่นยำ 80% แล้ว

รายละเอียดเพิ่มเติมอยู่ใน [YouTube CC dataset guide](docs/youtube_cc_dataset.md)

## Analyze Workflow

1. ผู้ใช้อัปโหลดวิดีโอไม่เกิน 5 นาที
2. Backend สร้าง `job_id`; Flutter polling `/jobs/{job_id}`
3. Pipeline แยกเสียงและถอดเสียงไทย/อังกฤษด้วย Faster Whisper
4. สกัด content keywords และ hook keywords
5. Classification เปรียบเทียบ transcript กับ centroid ของ train transcripts ที่ผ่าน human review
6. หากหมวดไม่พร้อมหรือ confidence ต่ำ ระบบคืน `Unknown/Other`
7. Recommendation เลือกคลิปผลงานสูงใน leaf เดียวกัน โดยใช้สถิติจริง ณ เวลา collect
8. คำนวณ keyword gap และ recommended duration จากแถวที่ใช้จริง
9. Result แสดง source, sample size, platform count, exemplar title และ warning เมื่อใช้ fallback

Classification ใช้ตัวอย่างทุกแถวด้วยน้ำหนักเท่ากัน เพื่อไม่ให้ช่องยอดนิยมครอบงำหมวด
Recommendation จึงค่อยเลือก top 40% จาก average views/day และ engagement rate
เพื่อใช้เป็นหลักฐานของคำแนะนำ

## Dashboard Workflow

- Background scheduler เก็บ live trends เป็น snapshot ทุก 60 วินาที
- Endpoint dashboard อ่าน snapshot ล่าสุดจาก DB/cache ไม่ออกอินเทอร์เน็ตระหว่าง request
- Provider ใดล้ม ระบบยังคืน HTTP 200 พร้อมข้อมูลจาก provider ที่สำเร็จ
- หลัง login snapshot แรกเป็น baseline; snapshot ถัดไปสร้าง `new_live_trend` notification
  เฉพาะ trend key ใหม่และไม่แจ้งซ้ำใน session เดิม

## Database Tables For Dataset

- `dataset_collection_runs`: รอบค้นหา, query config, artifact hash และสถานะ collector
- `dataset_contents`: transcript ที่ approve แล้ว พร้อม taxonomy, license, statistics และ split
- `dataset_review_events`: audit ของ approve/reject และผู้ตรวจ
- `taxonomy_nodes`: โครงสร้าง Level 1-3 และ readiness ของแต่ละ leaf
- `classification_models`: version และสถานะโมเดล
- `model_evaluation_metrics`: accuracy/F1 ทั้งภาพรวมและรายหมวด

## Current Limitation And Next Step

`youtube-transcript-api` ไม่ใช่ YouTube Data API อย่างเป็นทางการ จึงอาจถูกจำกัดหรือเปลี่ยนแปลงได้
ส่วน Captions API ทางการอนุญาตให้ดาวน์โหลด caption เมื่อมี OAuth permission ที่เหมาะสมกับวิดีโอนั้น
เท่านั้น หาก transcript ดึงไม่ได้ Collector จะข้ามคลิป ไม่สร้าง transcript ปลอมและไม่ใช้ filename fallback
กับ training dataset

หลังได้อย่างน้อย 30-50 reviewed clips ต่อหมวด ขั้นต่อไปคือ train/evaluate classifier โดยแยกตาม
Channel ID และเปิด active model เฉพาะเมื่อ macro F1 และผลรายหมวดผ่านเกณฑ์ที่กำหนด เช่น 0.80

## Tests

```powershell
python -m compileall -q app scripts tests
python -m unittest discover -s tests -v
cd frontend_flutter
flutter test
```
