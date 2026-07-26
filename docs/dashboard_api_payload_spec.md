# Dashboard API Payload Specification

This document describes the API payloads and response structures for frontend integration.

## Endpoints

### 1. GET /dashboard/overview

Description
- Returns dashboard overview metrics, distributions, live trend captures, and topic insights.
- Intended for the main admin/user dashboard.

Query parameters
- `region`: string, default from `settings.youtube_region` (e.g. `TH`). Must be 2 characters.
- `trend_mode`: string, one of `auto`, `mock`, `live`. Default `auto`.
- `trend_limit`: integer, default `5`, min `1`, max `10`.

Example request
```http
GET /dashboard/overview?region=TH&trend_mode=live&trend_limit=5 HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

Response structure
```json
{
  "db_status": "ok",
  "db_error": null,
  "user_role": "user",
  "metrics": {
    "total_users": 0,
    "active_users": 0,
    "total_user_contents": 0,
    "total_analysis_results": 0,
    "total_clusters": 0,
    "total_cluster_runs": 0,
    "total_cluster_memberships": 0,
    "total_dataset_contents": 0,
    "total_system_logs": 0,
    "my_contents": 0,
    "my_analysis_results": 0
  },
  "category_distribution": [
    { "category": "Technology", "count": 12 }
  ],
  "cluster_distribution": [
    { "cluster_name": "AudioTech", "count": 8 }
  ],
  "top_trends": [
    {
      "dataset_id": 1,
      "title": "Sample Trend Video",
      "category": "Technology",
      "source_platform": "youtube_live",
      "video_url": "https://www.youtube.com/watch?v=xxx",
      "views": 123456,
      "likes": 7890,
      "comments": 234,
      "trend_score": 45678.0,
      "published_at": "2026-07-26T14:00:00"
    }
  ],
  "top_categories": [
    { "category": "Technology", "count": 12 }
  ],
  "top_keywords": [
    { "keyword": "smartphone", "count": 125 }
  ],
  "source_distribution": [
    { "source_platform": "youtube_live", "count": 18 }
  ],
  "platform_summaries": [
    {
      "source": "youtube",
      "dataset_count": 42,
      "profile_count": 5,
      "domains": ["audio", "smartphone"]
    }
  ],
  "platform_comparison": [
    {
      "domain": "audio",
      "youtube_sample_size": 20,
      "google_sample_size": 12,
      "youtube_duration": "60-90 sec",
      "google_duration": "55-85 sec"
    }
  ],
  "priority_topics": [
    { "keyword": "content", "score": 12.34 }
  ],
  "emerging_topics": [
    { "keyword": "unbox", "score": 9.87 }
  ],
  "priority_items": [
    {
      "title": "Affordable smartphone camera battle under budget",
      "category": "28",
      "source_platform": "youtube_mock_th",
      "video_url": "https://www.youtube.com/watch?v=mock002",
      "trend_score": 418275.0,
      "priority_score": 419005.0,
      "novelty_score": 6.0,
      "views": 618000,
      "likes": 50100,
      "comments": 3025,
      "published_at": "2026-04-27T12:30:00"
    }
  ],
  "youtube_trends": {
    "mode": "mock",
    "region": "TH",
    "total": 5,
    "items": [ /* YouTubeTrendItem[] */ ]
  },
  "google_trends": {
    "mode": "mock",
    "region": "TH",
    "total": 5,
    "items": [ /* GoogleTrendItem[] */ ]
  }
}
```

Notes for frontend
- Use `priority_items` for an item-level priority ranking panel.
- Use `priority_topics` to display keyword-level priority signals.
- Use `emerging_topics` to show trending keywords with low historical frequency.
- `youtube_trends` and `google_trends` are useful for a trends summary card or chart.

---

### 2. GET /dashboard/emerging-topics

Description
- Returns trend topic insights and priority-ranked trend items.
- Recommended for a dedicated "Emerging Topics" UI screen.

Query parameters
- `region`: string, 2 letters.
- `trend_mode`: `auto`, `mock`, `live`.
- `trend_limit`: integer 1-10.

Example request
```http
GET /dashboard/emerging-topics?region=TH&trend_mode=live&trend_limit=5 HTTP/1.1
Host: localhost:8000
Authorization: Bearer <token>
```

Response structure
```json
{
  "priority_items": [
    {
      "title": "Affordable smartphone camera battle under budget",
      "category": "28",
      "source_platform": "youtube_mock_th",
      "video_url": "https://www.youtube.com/watch?v=mock002",
      "trend_score": 418275.0,
      "priority_score": 419005.0,
      "novelty_score": 6.0,
      "views": 618000,
      "likes": 50100,
      "comments": 3025,
      "published_at": "2026-04-27T12:30:00"
    }
  ],
  "emerging_topics": [
    { "keyword": "content", "score": 9.87 }
  ],
  "youtube_trends": {
    "mode": "mock",
    "region": "TH",
    "total": 5,
    "items": [ /* YouTubeTrendItem[] */ ]
  },
  "google_trends": {
    "mode": "mock",
    "region": "TH",
    "total": 5,
    "items": [ /* GoogleTrendItem[] */ ]
  }
}
```

Recommendation for frontend
- Use `priority_items` as the main ordered list of high-priority trend clips.
- Use `emerging_topics` as a short keyword recommendation list.
- Optionally show an alert or badge when `youtube_trends.mode` or `google_trends.mode` is `live`.

---

## Why Postman / curl scripts matter

Purpose
- They are tools for testing and validating the backend API before frontend integration.
- They let developers make real HTTP requests and inspect the exact JSON response.

Benefits
- verifies that the API endpoint works and returns the expected payload shape
- catches mismatches between frontend assumptions and backend output
- useful for debugging authentication, query parameters, and response format
- can be used in CI, shared with teammates, or imported by non-developers

Example curl script
```bash
curl -X GET "http://localhost:8000/dashboard/emerging-topics?region=TH&trend_mode=live&trend_limit=5" \
  -H "Authorization: Bearer <token>" \
  -H "Accept: application/json"
```

Example Postman usage
- import the request in Postman
- set the `Authorization` header or Bearer token
- send the request and inspect the JSON body
- save the request for repeated regression testing

Use case
- frontend developers can use it to confirm the payload fields before writing UI components
- QA testers can verify endpoint behavior manually
- backend developers can quickly verify live vs mock data without building the UI
