# YouTube Creative Commons Transcript Dataset

## Objective

สร้างชุดข้อมูลจริงสำหรับ Classification และ Recommendation โดยใช้วิดีโอ YouTube
ที่ผู้เผยแพร่เลือก Creative Commons, transcript จริงจาก public caption และ label ที่มนุษย์ตรวจ
ไม่ใช้ generated transcript, generated label หรือ demo seed เป็น production evidence

## Data Flow

```text
Project taxonomy queries
        |
        v
YouTube search.list
  type=video
  videoLicense=creativeCommon
  videoCaption=closedCaption
        |
        v
YouTube videos.list
  snippet, contentDetails, statistics, status
        |
        +-- reject: license is not creativeCommon
        +-- reject: duration is missing or invalid
        +-- reject: no supported public transcript
        |
        v
first 300 transcript seconds + candidates.jsonl + SHA-256 manifest
        |
        v
Human review: video + transcript + Level 3 label
        |
        +-- reject -> dataset_review_events only
        +-- approve -> dataset_contents + dataset_review_events
        |
        v
Channel-grouped train / validation / test
```

หน้า Admin `/#/admin-dataset-review` เป็น review workflow หลัก ผู้ตรวจเปิดวิดีโอ อ่าน
transcript เลือกหมวดและคุณภาพ แล้วกด Approve/Reject ได้โดยไม่แก้ CSV เอง ชื่อผู้ตรวจและเวลา
มาจากบัญชี Admin ที่ login อยู่ ส่วน CSV ยังคงเป็นหลักฐานและช่องทาง offline fallback

## Why Human Review Is Required

คำค้น เช่น `รีวิวมือถือ` บอกเพียงเหตุผลที่ระบบพบวิดีโอ ไม่ยืนยันว่าคลิปนั้นเป็น Phone จริง
Reviewer จึงต้องดูคลิปหรืออ่าน transcript แล้วตัดสิน Level 3 label เอง Search query ถูกเก็บไว้
เพื่อ audit แต่ไม่ถูกใช้เป็น label ที่ผ่านการตรวจ

## Review Rules

Approve เมื่อครบทุกข้อ:

1. ลิงก์เปิดได้และตรงกับ metadata
2. วิดีโอมี Creative Commons ณ เวลาที่ collect
3. เนื้อหาหลักอยู่ใน taxonomy leaf ที่เลือก
4. Transcript สะท้อนเสียงในคลิปและมีข้อมูลพอสำหรับจำแนก
5. Duration ของวิดีโอต้นทางตรวจสอบได้และมากกว่า 0 วินาที
6. เนื้อหาในช่วง transcript 300 วินาทีแรกชัดพอที่จะตรวจ label
7. ระบุ reviewer และเวลาตรวจจริง

ใช้ `transcript_quality=good` เมื่อ transcript ถูกต้องชัดเจน และ `acceptable` เมื่อมีความผิดพลาด
เล็กน้อยแต่ยังรักษาความหมายสำคัญ Reject เมื่อ transcript ผิดคลิป, ว่าง, อ่านไม่ได้ หรือหมวดไม่ชัด

## Production Eligibility

`app/services/dataset_eligibility.py` เป็น contract กลาง Classification, Recommendation และ
readiness coverage ใช้ query เดียวกัน จึงไม่มี endpoint ใดดึง candidate ที่ยังไม่ตรวจไปใช้ได้

ค่าหลักของแถว production:

```text
dataset_source      = youtube_cc
verification_status = human_verified
label_source        = human_review
transcript_source   = youtube_public_caption
taxonomy_version    = content-taxonomy-v1
is_training_eligible = true
is_keyword_recommendation_eligible = true
is_duration_recommendation_eligible = true/false ตาม source duration
is_active            = true
```

ความยาววิดีโอต้นทางไม่ถูกจำกัดที่ 5 นาที เพราะ source ที่ยาวกว่านั้นยังเป็นตัวอย่างจริงสำหรับ
Classification และ Keyword Recommendation ได้ อย่างไรก็ตาม ระบบเก็บ transcript เฉพาะช่วง
300 วินาทีแรกเพื่อให้ distribution ตรงกับ input ผู้ใช้ และกำหนด
`is_duration_recommendation_eligible=true` เฉพาะ source ที่ยาวไม่เกิน 300 วินาที เพื่อไม่ให้
วิดีโอแบบ long-form บิดคำแนะนำความยาวของ short review

## Split And Leakage Control

ระบบ hash `source_channel_id` แล้วกำหนด split แบบคงที่ 70/15/15 วิดีโอจากช่องเดียวกันจึงอยู่
split เดียวกันทั้งหมด วิธีนี้ลดโอกาสที่สำนวนหรือชื่อสินค้าจาก creator เดิมจะอยู่ทั้ง train และ test

## Recommendation Evidence

### Bulk collection strategies

Each leaf has two explicit candidate sampling targets:

1. `recommendation_high_performance` uses YouTube Search `order=viewCount` first.
2. `classification_diverse` uses `order=relevance` to fill the remaining target.

Candidate rows store the discovery strategy, lifetime statistics captured at
collection time, average views per day, engagement rate, and a reproducible
performance signal. Human approval is still required before either strategy can
enter the production dataset.

The collector checkpoints JSONL, review CSV, manifest, hashes, counters, and
the next page token after each successful search page. Resume uses the same run
ID and page tokens. It is blocked after the first review event so reviewed
artifacts cannot change. New runs deduplicate against both approved database
rows and all registered candidate artifacts by YouTube ID and transcript hash.

Collection schema version 4 adds two dataset quality gates:

1. Reserve at least 40% Thai transcripts per leaf by default (`20/50`). English
   candidates are skipped after the non-Thai capacity is full; transcripts are
   never machine-translated to satisfy this target.
2. Accept at most 3 videos from one Channel ID in one leaf, counting prior
   production rows and registered candidate artifacts, to reduce creator leakage.

Phone and Camera discovery queries include stronger product/context terms and
negative terms for common false positives. Search queries remain discovery hints;
only human review assigns the final label. The manifest, CLI checkpoint output,
and Admin Dataset Review page report progress by leaf, transcript language, and
unique channel count.

Classification ใช้ train rows ทุกแถวในหมวดที่พร้อมด้วยน้ำหนักเท่ากัน ส่วน Recommendation:

1. Query เฉพาะ taxonomy leaf เดียวกับคลิปผู้ใช้
2. จัดอันดับด้วย performance signal จาก average views/day และ engagement rate ณ เวลา collect
3. เลือก top 40% อย่างน้อย 10 รายการ หาก pool มีเพียงพอ
4. สกัด keyword frequency จากทุกรายการที่เลือกจริง แต่คำนวณ duration เฉพาะแถวที่
   `is_duration_recommendation_eligible=true`
5. ส่ง dataset row IDs, YouTube IDs, exemplar titles และ sample counts ไปกับ evidence

ค่า statistics เป็น snapshot ณ เวลา collect ไม่ใช่ยอดปัจจุบันแบบ live

## Collector Resume And Quota

Collector บันทึก checkpoint หลังแต่ละ search page หาก YouTube ตอบ HTTP 429 ระบบบันทึกสถานะ
`quota_waiting` แทน `failed` และไม่นับเป็น data error จากนั้น resume run เดิมได้เมื่อ quota
พร้อมอีกครั้ง อย่างไรก็ตาม run ที่สร้างก่อน schema version 3 เคยตัด source video ที่ยาว
เกิน 300 วินาทีทิ้ง จึงห้าม resume ภายใต้นโยบายใหม่ ให้เริ่ม collection run ใหม่แทน ระบบจะ
deduplicate YouTube ID และ transcript hash กับ 35 candidates เดิมโดยอัตโนมัติ

## Reproducibility

เก็บ SHA-256 ของ candidate JSONL, review CSV, transcript และ import batch ทุกครั้ง
หากมีการแก้ artifact หลัง collect/import hash จะไม่ตรงและ importer ปฏิเสธข้อมูล

## Classification Training And Evaluation

`scripts/train_classification_models.py` อ่านเฉพาะแถว production ที่ผ่าน human
review จาก `dataset_contents` และตรวจ split จาก Channel ID ซ้ำอีกครั้งก่อน train
หากช่องเดียวกันรั่วข้าม split หรือหมวดใดไม่มี train/validation/test ระบบคืน
`not_ready` และไม่สร้าง model row

Dataset artifacts ประกอบด้วย `train.jsonl`, `validation.jsonl`, `test.jsonl`,
SHA-256 และ manifest ที่ระบุ Dataset version, taxonomy version, ภาษา, จำนวนช่อง
และจำนวนตัวอย่างรายหมวด โมเดลที่ benchmark มีสามแบบ:

1. Word/character TF-IDF + balanced Logistic Regression
2. Word/character TF-IDF + Complement Naive Bayes
3. Word/character TF-IDF + balanced SGD Logistic Regression

ระบบคำนวณ Accuracy, Macro Precision, Macro Recall, Macro F1, per-class metrics
และ Confusion Matrix สำหรับ validation/test พร้อมมุมมองแยกภาษาไทย/อังกฤษ
`Unknown/Other` ใช้ confidence rejection (`0.60` โดยค่าเริ่มต้น) จึงไม่เพิ่มข้อมูล
สังเคราะห์หรือ label ปลอมเข้า training set

Promotion gate กำหนดให้ validation/test Accuracy และ Macro F1 ต้องผ่าน `0.80`
ทุกค่า โมเดลที่ผ่านถูกบันทึกสถานะ `qualified`; โมเดลที่ไม่ผ่านเป็น
`evaluated_below_threshold` ทั้งสองสถานะยังมี `is_active=false` จนกว่าจะเชื่อม
artifact เข้ากับ runtime และ promote อย่างชัดเจน

ก่อน Dataset พร้อม ใช้ `--smoke-test` เพื่อตรวจเส้นทาง train, metric persistence,
artifact reload และ prediction ได้ โหมดนี้ใช้เฉพาะข้อมูล human-reviewed จริงที่มี
อยู่ แต่ติดป้าย `smoke_test_only`, บล็อก promotion และห้ามอ้างว่าเป็นโมเดลสุดท้าย

## Source Notes

- YouTube Search API รองรับ `videoLicense=creativeCommon` และ `videoCaption=closedCaption`:
  https://developers.google.com/youtube/v3/docs/search/list
- Videos API ให้ duration, statistics และ `status.license`:
  https://developers.google.com/youtube/v3/docs/videos
- YouTube อธิบาย Creative Commons Attribution ที่:
  https://support.google.com/youtube/answer/2797468
- Official Captions API ต้องใช้สิทธิ์ OAuth ที่เหมาะสมสำหรับการดาวน์โหลด caption:
  https://developers.google.com/youtube/v3/guides/implementation/captions
- Public transcript collector ที่ใช้เป็น unofficial dependency:
  https://github.com/jdepoix/youtube-transcript-api

## Roadmap

1. Collect candidate มากกว่าจำนวนเป้าหมายประมาณ 1.5-2 เท่า เผื่อ reject
2. Human review อย่างน้อย 30-50 approved clips ต่อ leaf
3. ตรวจ class balance, language balance, creator distribution และ duplicate report
4. Train baseline classifier และบันทึก model version/metrics
5. เปรียบเทียบ candidate models ด้วย macro F1 และ per-class recall
6. เปิด active model เมื่อ test set ผ่านเกณฑ์ 0.80 ที่ตกลงกับอาจารย์
7. เพิ่มหมวดย่อยใหม่เฉพาะเมื่อมีข้อมูลจริงและ evaluation เพียงพอ
