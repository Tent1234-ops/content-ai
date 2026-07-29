# Content Trend Analysis API Reference

เอกสารนี้เป็น API reference สั้น ๆ สำหรับ backend ของระบบวิเคราะห์แนวโน้มคอนเทนต์

## Authentication

ทุก endpoint ที่ต้องการสิทธิ์ใช้งานต้องส่ง header:

    Authorization: Bearer <access_token>

สำหรับแอดมินให้ใช้:

    Authorization: Bearer <admin_token>


ทุก endpoint ที่ต้องการสิทธิ์ใช้งานต้องส่ง header:

    Authorization: Bearer <access_token>

### POST /auth/register
ลงทะเบียนผู้ใช้งานใหม่

Request (JSON):

    {
      "username": "creator_demo",
      "email": "creator_demo@example.com",
      "password": "password123"
    }

Admin registration:

    {
      "username": "admin_demo",
      "email": "admin_demo@example.com",
      "password": "password123",
      "role": "admin",
      "admin_invite_code": "<ADMIN_INVITE_CODE>"
    }

Response:

    {
      "user_id": 1,
      "username": "creator_demo",
      "email": "creator_demo@example.com",
      "role": "user",
      "is_active": true,
      "created_at": "2026-07-28T12:34:56"
    }

### POST /auth/login
ล็อกอินรับ JWT token

Request (JSON):

    {
      "email": "creator_demo@example.com",
      "password": "password123"
    }

Response:

    {
      "access_token": "<jwt-token>",
      "token_type": "bearer",
      "user": { ... }
    }

### GET /auth/me
ดูข้อมูลผู้ใช้งานปัจจุบัน

Headers: Authorization ต้องมี

Response fields: user_id, username, email, role, is_active, created_at

## Upload / Analysis

### POST /analyze
อัปโหลดวิดีโอแล้วกลับผลวิเคราะห์ + recommendation โดยไม่บันทึก

Form data:

    file: <video file>

Response fields:

    transcript, analysis, recommendation

### POST /analyze/save
อัปโหลดวิดีโอ วิเคราะห์ recommendation และบันทึกผลลงฐานข้อมูล

Form data:

    file: <video file>

Response fields:

    content_id, saved_keywords, recommended_keywords, recommended_duration,
    recommendation, analysis, nlp_result

## User Content / My Ideas

### GET /contents/my
ดึงประวัติการวิเคราะห์ของผู้ใช้งาน

Query:

    limit, offset

Response fields:

    total, items

### GET /contents/{content_id}
ดูรายละเอียดรายการวิดีโอที่บันทึกไว้

Response fields:

    content_id, title, created_at, video_url, transcript, analysis, nlp_result, recommendation

## Recommendation

### POST /recommendations/from-text
สร้าง recommendation จากข้อความ / ชื่อเรื่อง

Request (JSON):

    {
      "title": "รีวิวมือถือใหม่",
      "text": "รีวิวมือถือใหม่ ฟีเจอร์กล้องชัด แบตอึด",
      "source": "youtube",
      "profile_limit": 150
    }

### GET /recommendations/from-content/{content_id}
สร้าง recommendation จากเนื้อหาที่บันทึกไว้

Query:

    source=youtube|google
    profile_limit=10-500

### GET /recommendations/profiles
ดึงโปรไฟล์โดเมนจาก dataset

Query:

    source=youtube|google
    limit=10-500

### GET /recommendations/profiles/compare
เปรียบเทียบโปรไฟล์ระหว่างสองแหล่งข้อมูล

Query:

    left_source=youtube|google
    right_source=youtube|google
    limit=10-500

### GET /recommendations/admin/report
รายงาน recommendation profile สำหรับ admin

## Dashboard

### GET /dashboard/overview
ข้อมูลภาพรวม dashboard

Query:

    region=TH
    trend_mode=auto|mock|live
    trend_limit=1-10

Response fields:

    db_status, user_role, metrics, category_distribution,
    cluster_distribution, top_trends, top_categories, top_keywords,
    source_distribution, platform_summaries, platform_comparison,
    priority_topics, emerging_topics, priority_items,
    youtube_trends, google_trends, tiktok_trends

### GET /dashboard/emerging-topics
แสดงเทรนด์ priority และ emerging topics

Query:

    region, trend_mode, trend_limit

Response fields:

    priority_items, emerging_topics, youtube_trends, google_trends, tiktok_trends

## Trends

### GET /trends/youtube
รับรายการเทรนด์ YouTube

Query:

    limit, region, mode=auto|mock|live

### GET /trends/youtube/categories
รับหมวดหมู่ YouTube ที่รองรับ

### POST /trends/youtube/sync
ซิงก์เทรนด์ YouTube ลง dataset (admin)

### GET /trends/google
รับรายการเทรนด์ Google

Query:

    limit, region, mode=auto|mock|live

### POST /trends/google/sync
ซิงก์เทรนด์ Google ลง dataset (admin)

### GET /trends/tiktok
รับรายการเทรนด์ TikTok

Query:

    limit, region, mode=auto|mock|live

### POST /trends/tiktok/sync
ซิงก์เทรนด์ TikTok ลง dataset (admin)

## Follows / Notifications

### POST /follows/topic
ติดตามคำสำคัญหรือหมวดหมู่ใหม่

Request (JSON):

    {
      "match_type": "keyword",
      "value": "รีวิว"
    }

### GET /follows/topics
ดูรายการหัวข้อที่ติดตาม

### DELETE /follows/topic/{id}
ยกเลิกการติดตามหัวข้อ

### GET /notifications/
ดึงการแจ้งเตือนของผู้ใช้งาน

Query:

    unread_only, limit, offset

### POST /notifications/mark_read
ทำเครื่องหมายการแจ้งเตือนว่าอ่านแล้ว

Request (JSON):

    {
      "ids": [1, 2, 3]
    }

### POST /notifications/create_for_user
สร้าง notification ด้วย admin helper

## Classification / Clustering / NLP / Datasets

### POST /classification/text
จำแนกโดเมนจากข้อความ

### POST /clustering/kmeans
### POST /clustering/hdbscan
### POST /clustering/kmeans/save
### POST /clustering/hdbscan/save
### POST /clustering/from-dataset

### POST /nlp/extract
### POST /nlp/extract/save

### GET /datasets/youtube
### GET /datasets/google

## Admin

### GET /admin/me
ดูข้อมูล admin ปัจจุบัน

### GET /admin/datasets
ดู dataset content ที่อยู่ในระบบ

### GET /admin/clusters/runs
ดูรายการการรัน clustering

### GET /admin/clusters/runs/{run_id}
ดูรายละเอียดการรัน clustering

### GET /admin/logs
ดู system log

### GET /admin/reports/overview
ดูรายงานภาพรวม admin

### GET /admin/settings
ดึงค่า configuration ปัจจุบัน

### PUT /admin/settings
แก้ไขค่า configuration

Request (JSON):

    {
      "max_keywords_display": 15,
      "hook_analysis_duration": 90,
      "analysis_time_range_days": 120,
      "youtube_region": "TH",
      "google_region": "TH",
      "tiktok_region": "TH",
      "enable_youtube_trending": true,
      "enable_google_trends": true,
      "enable_tiktok_trending": true,
      "auto_scan_interval_hours": 6
    }

### POST /admin/settings/validate
ตรวจสอบ config ก่อนบันทึก

### POST /admin/settings/reset?confirm=true
ตั้งค่ากลับเป็นค่าเริ่มต้น

### GET /admin/statistics
สถิติ admin dashboard

### POST /admin/settings/backup
สำรอง config เป็น JSON

### POST /admin/settings/restore?confirm=true
กู้ config จาก backup

### POST /admin/settings/audit-log
ดู audit log การเปลี่ยนแปลง config

### POST /admin/health
ตรวจสอบสุขภาพระบบ

### POST /admin/scan/youtube
### POST /admin/scan/google
### POST /admin/scan/tiktok

สรุป: เอกสารนี้เป็น reference ครบทุก endpoint ที่มีใน backend และใช้ `Authorization: Bearer <token>` สำหรับ route ที่ป้องกันด้วย JWT.