import time
import unittest
from datetime import timedelta
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.core.security import decode_access_token, hash_password
from app.database.models import (
    Notification,
    TrendSnapshotRun,
    User,
    UserTrendWatchSession,
)
from app.routes.auth import login
from app.schemas.trends import GoogleTrendItem, YouTubeTrendItem
from app.schemas.auth import LoginRequest
from app.schemas.notifications import NotificationItem
from app.routes.dashboard import dashboard_live_trends_snapshot
from app.services.dashboard import build_dashboard_summary
from app.services.live_trend_notifications import compare_live_trend_snapshot
from app.services.live_trend_snapshots import (
    load_latest_live_snapshot,
    normalize_live_item,
    refresh_global_live_trends,
)
from app.services.trend_watch_sessions import start_trend_watch_session
from app.services.view_metrics import (
    YOUTUBE_PLAY_START_VIEW_V2,
    YOUTUBE_QUALIFIED_VIEW_V1,
)


class LiveTrendSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.user = User(
            username="snapshot-test",
            email="snapshot@example.com",
            password_hash=hash_password("test-password"),
            role="user",
        )
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    @staticmethod
    def _youtube(_region, _limit):
        return "live", [
            YouTubeTrendItem(
                title="YouTube trend A",
                channel_title="Channel",
                category="Technology",
                video_url="https://youtube.example/a",
                views=1000,
                likes=100,
                comments=10,
                trend_score=730,
                source="youtube_live",
            )
        ]

    @staticmethod
    def _google(_region, _limit):
        return "live", [
            GoogleTrendItem(
                title="Google trend B",
                query="Google trend B",
                category="Search",
                video_url="https://google.example/b",
                views=0,
                likes=0,
                comments=0,
                trend_score=500,
                source="google_trends_live",
            )
        ]

    @staticmethod
    def _failed_tiktok(_region, _limit):
        raise TimeoutError("simulated TikTok failure")

    @staticmethod
    def _empty_provider(_region, _limit):
        return "live", []

    @staticmethod
    def _youtube_titles(*titles):
        def fetcher(_region, _limit):
            return "live", [
                YouTubeTrendItem(
                    title=title,
                    channel_title="Channel",
                    category="Technology",
                    video_url=f"https://youtube.example/{title.lower()}",
                    views=1000,
                    likes=100,
                    comments=10,
                    trend_score=730,
                    source="youtube_live",
                )
                for title in titles
            ]

        return fetcher

    @staticmethod
    def _youtube_metric(views, likes, comments):
        def fetcher(_region, _limit):
            return "live", [
                YouTubeTrendItem(
                    title="Keyboard momentum",
                    channel_title="Channel",
                    category="Technology",
                    video_url="https://youtube.example/keyboard-momentum",
                    views=views,
                    likes=likes,
                    comments=comments,
                    trend_score=730,
                    source="youtube_live",
                )
            ]

        return fetcher

    @staticmethod
    def _youtube_versioned_metric(views, version):
        def fetcher(_region, _limit):
            return "live", [
                {
                    "title": "Versioned momentum",
                    "category": "Technology",
                    "video_url": "https://youtube.example/versioned-momentum",
                    "views": views,
                    "likes": 100,
                    "comments": 10,
                    "trend_score": 730,
                    "source": "youtube_live",
                    "view_metric_version": version,
                }
            ]

        return fetcher

    @staticmethod
    def _google_ranked(*titles):
        def fetcher(_region, _limit):
            return "live", [
                GoogleTrendItem(
                    title=title,
                    query=title,
                    category="TH",
                    video_url=f"https://google.example/{title.lower()}",
                    trend_score=float(50_000 - (index * 10_000)),
                    source="google_trends_live",
                )
                for index, title in enumerate(titles)
            ]

        return fetcher

    def _start_watch_session(self):
        return start_trend_watch_session(self.db, user=self.user, region="TH")

    def _backdate_latest_run(self, *, minutes=6):
        run = (
            self.db.query(TrendSnapshotRun)
            .order_by(TrendSnapshotRun.run_id.desc())
            .first()
        )
        run.started_at -= timedelta(minutes=minutes)
        if run.completed_at is not None:
            run.completed_at -= timedelta(minutes=minutes)
        self.db.commit()

    def test_partial_run_keeps_successful_platforms_and_reads_quickly(self):
        result = refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube,
                "google": self._google,
                "tiktok": self._failed_tiktok,
            },
            db=self.db,
        )

        started = time.perf_counter()
        snapshot = load_latest_live_snapshot(self.db, region="TH", limit=50)
        elapsed = time.perf_counter() - started

        self.assertEqual(result["status"], "partial")
        self.assertEqual(snapshot["platforms"]["youtube"]["total"], 1)
        self.assertEqual(snapshot["platforms"]["google"]["total"], 1)
        self.assertEqual(snapshot["platforms"]["tiktok"]["mode"], "live_error")
        self.assertTrue(result["completed_at"].endswith("Z"))
        self.assertTrue(snapshot["generated_at"].endswith("Z"))
        self.assertLess(elapsed, 1.0)

    def test_user_snapshot_uses_cached_run_as_baseline(self):
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube,
                "google": self._google,
                "tiktok": self._failed_tiktok,
            },
            db=self.db,
        )
        watch_session = self._start_watch_session()

        result = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )

        self.assertEqual(result["new_count"], 0)
        self.assertEqual(result["platforms"]["youtube"]["total"], 1)
        self.assertEqual(result["platforms"]["google"]["total"], 1)
        self.assertEqual(result["platforms"]["tiktok"]["mode"], "live_error")

    def test_dashboard_snapshot_endpoint_reads_cached_data_under_one_second(self):
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube,
                "google": self._google,
                "tiktok": self._failed_tiktok,
            },
            db=self.db,
        )
        watch_session = self._start_watch_session()

        started = time.perf_counter()
        result = dashboard_live_trends_snapshot(
            region="TH",
            trend_limit=50,
            current_user=self.user,
            watch_session=watch_session,
            db=self.db,
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(result["platforms"]["youtube"]["total"], 1)
        self.assertEqual(result["platforms"]["google"]["total"], 1)
        self.assertEqual(result["platforms"]["tiktok"]["mode"], "live_error")
        self.assertLess(elapsed, 1.0)

    def test_dashboard_summary_live_mode_never_calls_external_providers(self):
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube,
                "google": self._google,
                "tiktok": self._failed_tiktok,
            },
            db=self.db,
        )

        with (
            patch("app.services.dashboard.get_youtube_trending", side_effect=AssertionError("network call")),
            patch("app.services.dashboard.get_google_trending", side_effect=AssertionError("network call")),
            patch("app.services.dashboard.get_tiktok_trending", side_effect=AssertionError("network call")),
        ):
            started = time.perf_counter()
            result = build_dashboard_summary(
                db=self.db,
                current_user=self.user,
                region="TH",
                trend_mode="live",
                trend_limit=50,
            )
            elapsed = time.perf_counter() - started

        self.assertEqual(result["youtube_trends"]["total"], 1)
        self.assertEqual(result["google_trends"]["total"], 1)
        self.assertEqual(result["tiktok_trends"]["mode"], "live_error")
        self.assertLess(elapsed, 1.0)

    def test_login_uses_current_snapshot_as_baseline(self):
        snapshot = refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube,
                "google": self._google,
                "tiktok": self._failed_tiktok,
            },
            db=self.db,
        )

        response = login(
            LoginRequest(email=self.user.email, password="test-password"),
            db=self.db,
        )
        watch_session = (
            self.db.query(UserTrendWatchSession)
            .filter(UserTrendWatchSession.session_key == response.session_key)
            .one()
        )

        self.assertEqual(watch_session.baseline_run_id, snapshot["run_id"])
        self.assertEqual(watch_session.last_seen_run_id, snapshot["run_id"])
        self.assertEqual(decode_access_token(response.access_token)["sid"], response.session_key)
        self.assertEqual(self.db.query(Notification).count(), 0)

    def test_new_trend_is_notified_once_per_login_session(self):
        base_fetchers = {
            "youtube": self._youtube_titles("A", "B", "C"),
            "google": self._empty_provider,
            "tiktok": self._empty_provider,
        }
        refresh_global_live_trends(region="TH", limit=50, fetchers=base_fetchers, db=self.db)
        watch_session = self._start_watch_session()

        baseline = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        self.assertEqual(baseline["new_count"], 0)

        refresh_global_live_trends(region="TH", limit=50, fetchers=base_fetchers, db=self.db)
        unchanged = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        self.assertEqual(unchanged["new_count"], 0)

        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                **base_fetchers,
                "youtube": self._youtube_titles("A", "B", "C", "D"),
            },
            db=self.db,
        )
        changed = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        self.assertEqual(changed["new_count"], 1)
        self.assertEqual(changed["new_notifications"][0].title, "D")

        duplicate_poll = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        notifications = self.db.query(Notification).all()
        self.assertEqual(duplicate_poll["new_count"], 0)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].type, "new_live_trend")
        self.assertNotIn("Analysis complete", notifications[0].title)
        serialized = NotificationItem.model_validate(notifications[0]).model_dump(
            mode="json"
        )
        self.assertTrue(serialized["detected_at"].endswith("Z"))
        self.assertTrue(serialized["created_at"].endswith("Z"))

    def test_youtube_momentum_uses_stable_window_and_qualitative_label(self):
        empty = self._empty_provider
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_metric(1_000, 100, 10),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )
        self._backdate_latest_run()
        watch_session = self._start_watch_session()
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_metric(1_600, 160, 16),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )

        result = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        item = result["platforms"]["youtube"]["items"][0]

        self.assertEqual(item["change_kind"], "velocity_up")
        self.assertEqual(item["change_label"], "Gaining fastest")
        self.assertGreater(item["engagement_rate_per_minute"], 0)
        self.assertGreaterEqual(item["comparison_window_seconds"], 300)
        self.assertTrue(item["is_meaningful_rising"])
        self.assertNotIn("/min", item["change_label"])

    def test_youtube_momentum_ignores_single_refresh_noise(self):
        empty = self._empty_provider
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_metric(1_000, 100, 10),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )
        watch_session = self._start_watch_session()
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_metric(1_001, 101, 10),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )

        result = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        item = result["platforms"]["youtube"]["items"][0]

        self.assertEqual(item["change_kind"], "baseline")
        self.assertEqual(item["change_label"], "Baseline")
        self.assertEqual(item["engagement_rate_per_minute"], 0)
        self.assertFalse(item["is_meaningful_rising"])

    def test_youtube_momentum_resets_when_view_metric_version_changes(self):
        empty = self._empty_provider
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_versioned_metric(
                    1_000,
                    YOUTUBE_QUALIFIED_VIEW_V1,
                ),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )
        self._backdate_latest_run()
        watch_session = self._start_watch_session()
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": self._youtube_versioned_metric(
                    50_000,
                    YOUTUBE_PLAY_START_VIEW_V2,
                ),
                "google": empty,
                "tiktok": empty,
            },
            db=self.db,
        )

        result = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        item = result["platforms"]["youtube"]["items"][0]

        self.assertEqual(item["change_kind"], "metric_baseline")
        self.assertEqual(item["engagement_delta"], 0)
        self.assertEqual(item["engagement_rate_per_minute"], 0)
        self.assertFalse(item["has_previous_snapshot"])
        self.assertTrue(item["metric_version_changed"])

    def test_google_momentum_uses_rank_movement(self):
        empty = self._empty_provider
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": empty,
                "google": self._google_ranked("A", "B", "C"),
                "tiktok": empty,
            },
            db=self.db,
        )
        self._backdate_latest_run()
        watch_session = self._start_watch_session()
        refresh_global_live_trends(
            region="TH",
            limit=50,
            fetchers={
                "youtube": empty,
                "google": self._google_ranked("C", "A", "B"),
                "tiktok": empty,
            },
            db=self.db,
        )

        result = compare_live_trend_snapshot(
            db=self.db,
            user=self.user,
            watch_session=watch_session,
            region="TH",
            limit=50,
        )
        item = result["platforms"]["google"]["items"][0]

        self.assertEqual(item["title"], "C")
        self.assertEqual(item["rank_change"], 2)
        self.assertEqual(item["change_kind"], "rank_up")
        self.assertEqual(item["change_label"], "Up 2 ranks")
        self.assertTrue(item["is_meaningful_rising"])

    def test_google_region_is_not_exposed_as_a_content_category(self):
        normalized = normalize_live_item(
            "google",
            GoogleTrendItem(
                title="Thai search topic",
                query="Thai search topic",
                category="TH",
                video_url="https://google.example/topic",
                trend_score=20_000,
                source="google_trends_live",
            ),
        )

        self.assertEqual(normalized["category"], "Search Trends")
        self.assertEqual(normalized["views"], 0)


if __name__ == "__main__":
    unittest.main()
