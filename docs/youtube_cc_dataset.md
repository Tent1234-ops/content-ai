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
        +-- reject: duration > 300 seconds
        +-- reject: no supported public transcript
        |
        v
candidates.jsonl + SHA-256 manifest
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
5. ความยาวไม่เกิน 300 วินาที
6. ระบุ reviewer และเวลาตรวจจริง

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
is_active            = true
```

## Split And Leakage Control

ระบบ hash `source_channel_id` แล้วกำหนด split แบบคงที่ 70/15/15 วิดีโอจากช่องเดียวกันจึงอยู่
split เดียวกันทั้งหมด วิธีนี้ลดโอกาสที่สำนวนหรือชื่อสินค้าจาก creator เดิมจะอยู่ทั้ง train และ test

## Recommendation Evidence

Classification ใช้ train rows ทุกแถวในหมวดที่พร้อมด้วยน้ำหนักเท่ากัน ส่วน Recommendation:

1. Query เฉพาะ taxonomy leaf เดียวกับคลิปผู้ใช้
2. จัดอันดับด้วย performance signal จาก average views/day และ engagement rate ณ เวลา collect
3. เลือก top 40% อย่างน้อย 10 รายการ หาก pool มีเพียงพอ
4. สกัด keyword frequency และ duration จากรายการที่เลือกจริง
5. ส่ง dataset row IDs, YouTube IDs, exemplar titles และ sample counts ไปกับ evidence

ค่า statistics เป็น snapshot ณ เวลา collect ไม่ใช่ยอดปัจจุบันแบบ live

## Reproducibility

เก็บ SHA-256 ของ candidate JSONL, review CSV, transcript และ import batch ทุกครั้ง
หากมีการแก้ artifact หลัง collect/import hash จะไม่ตรงและ importer ปฏิเสธข้อมูล

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
