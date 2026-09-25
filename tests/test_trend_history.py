import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base, get_db
from app.database.models import TrendHistoryAttempt, TrendHistoryBucket, TrendSnapshotItem, TrendSnapshotRun
from app.routes import dashboard
from app.services.trend_history import (
    archive_snapshot_run, backfill_retained_history, load_trend_history, prune_trend_history,
)


class TrendHistoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.now = datetime(2026, 9, 19, 12, 30)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def run_sample(self, at, items=(('video-a', 1, 'Gaming'),), platform='youtube',
                   scope='global', status='ok', mode='live', region='TH'):
        provider = {'status': status, 'mode': mode}
        if scope != 'global':
            provider['categories'] = {scope.split(':')[1]: {'status': status}}
        run = TrendSnapshotRun(region=region,
            snapshot_kind='global' if scope == 'global' else 'youtube_categories',
            status='partial' if status == 'error' else 'completed',
            provider_status=json.dumps({platform: provider}), started_at=at, completed_at=at)
        run.items = [TrendSnapshotItem(platform=platform, ranking_scope=scope,
            trend_key=key, provider_rank=rank, title=key, category=category,
            source_platform=platform) for key, rank, category in items]
        self.db.add(run)
        self.db.flush()
        archive_snapshot_run(self.db, run)
        self.db.commit()
        return run

    def history(self, **kwargs):
        return load_trend_history(self.db, region='TH', platform='youtube', now=self.now, **kwargs)

    def test_hour_keeps_first_and_last_real_time_not_average_rank(self):
        self.run_sample(self.now - timedelta(minutes=20), (('a', 8, 'Gaming'),))
        self.run_sample(self.now - timedelta(minutes=10), (('a', 2, 'Gaming'),))
        self.run_sample(self.now, (('a', 4, 'Gaming'),))
        points = self.history()['points']
        self.assertEqual([p['ranks']['a'] for p in points], [8, 4])
        self.assertEqual(self.db.query(TrendHistoryBucket).count(), 1)
        self.assertTrue(points[0]['observed_at'].endswith('Z'))

    def test_backfill_idempotent_and_survives_deleting_raw_snapshots(self):
        run = self.run_sample(self.now)
        backfill_retained_history(self.db, now=self.now)
        backfill_retained_history(self.db, now=self.now)
        self.assertEqual(self.db.query(TrendHistoryBucket).count(), 1)
        self.db.delete(run)
        self.db.commit()
        self.assertEqual(self.history()['coverage']['sample_count'], 1)

    def test_regions_platforms_and_ranking_scopes_never_mix(self):
        self.run_sample(self.now, (('global', 3, 'Gaming'),))
        self.run_sample(self.now, (('category', 1, 'Gaming'),), scope='category:20')
        self.run_sample(self.now, (('search', 2, 'Search'),), platform='google')
        self.run_sample(self.now, (('foreign', 1, 'Gaming'),), region='US')
        self.assertEqual(self.history()['points'][0]['ranks'], {'global': 3})
        category = self.history(category_id='20')['points'][0]
        self.assertEqual(category['ranks'], {'category': 1})
        self.assertEqual(category['category_counts'], {})

    def test_provider_failure_or_mock_not_recorded_as_zero(self):
        self.run_sample(self.now - timedelta(minutes=20))
        self.run_sample(self.now, (), status='error')
        self.run_sample(self.now, (), mode='mock')
        self.assertEqual(self.history()['coverage']['sample_count'], 1)
        self.assertEqual(self.history()['points'][0]['total'], 1)

    def test_confirmed_empty_response_is_distinct_from_fetch_failure(self):
        self.run_sample(self.now - timedelta(minutes=20))
        self.run_sample(self.now, (), status='empty')
        points = self.history()['points']
        self.assertEqual(len(points), 2)
        self.assertEqual(points[-1]['ranks'], {})
        self.assertEqual(points[-1]['total'], 0)

    def test_category_mock_is_excluded_even_with_live_parent_metadata(self):
        run = TrendSnapshotRun(region='TH', snapshot_kind='youtube_categories',
            status='completed', started_at=self.now, completed_at=self.now,
            provider_status=json.dumps({'youtube': {'mode': 'live', 'status': 'ok',
                'categories': {'20': {'status': 'ok', 'mode': 'mock'}}}}))
        self.db.add(run)
        archive_snapshot_run(self.db, run)
        self.db.commit()
        self.assertEqual(self.history(category_id='20')['points'], [])

    def test_missing_video_has_no_invented_rank_and_gap_is_marked(self):
        self.run_sample(self.now - timedelta(days=2))
        self.run_sample(self.now, (('different', 1, 'Music'),))
        data = self.history()
        self.assertTrue(data['points'][1]['break_before'])
        self.assertNotIn('video-a', data['points'][1]['ranks'])
        old = next(i for i in data['items'] if i['key'] == 'video-a')
        self.assertIsNone(old['latest_rank'])
        self.assertEqual(data['coverage']['gap_count'], 1)

    def test_top_fifty_actual_denominator_not_category_collection_limit(self):
        self.run_sample(self.now, (('a', 1, 'Gaming'), ('b', 2, 'Music'), ('c', 51, 'Gaming')))
        point = self.history()['points'][0]
        self.assertEqual(point['total'], 2)
        self.assertEqual(point['category_counts'], {'Gaming': 1, 'Music': 1})

    def test_window_includes_only_real_observation_times_not_bucket_start(self):
        self.run_sample(self.now - timedelta(days=1, minutes=1))
        self.run_sample(self.now - timedelta(hours=23))
        self.assertEqual(self.history(days=1)['coverage']['sample_count'], 1)
        self.assertTrue(self.history(days=1)['coverage']['is_stale'])

    def test_prune_removes_only_expired_archive(self):
        self.run_sample(self.now - timedelta(days=91))
        self.run_sample(self.now)
        prune_trend_history(self.db, now=self.now)
        self.db.commit()
        self.assertEqual(self.db.query(TrendHistoryBucket).count(), 1)
        self.assertEqual(self.db.query(TrendSnapshotRun).count(), 2)

    def test_empty_history_does_not_generate_points(self):
        data = self.history()
        self.assertEqual(data['points'], [])
        self.assertEqual(data['items'], [])
        self.assertTrue(data['coverage']['is_stale'])

    def test_failure_survives_raw_prune_and_breaks_line_inside_one_hour(self):
        self.run_sample(self.now - timedelta(minutes=20))
        failed = self.run_sample(self.now - timedelta(minutes=10), (), status='error')
        self.run_sample(self.now)
        self.db.delete(failed)
        self.db.commit()
        data = self.history()
        self.assertEqual(data['coverage']['failed_attempts'], 1)
        self.assertTrue(data['points'][-1]['break_before'])
        hour = data['hours'][-1]
        self.assertEqual(hour['status'], 'partial')
        self.assertIsNone(self.db.query(TrendHistoryAttempt).filter_by(status='failed').one().sample_count)

    def test_long_term_archive_distinguishes_unobserved_from_failed(self):
        self.run_sample(self.now - timedelta(days=60))
        self.run_sample(self.now, (), status='error')
        data = self.history(days=90)
        self.assertEqual(data['coverage']['sample_count'], 1)
        self.assertTrue(any(h['status'] == 'no_observation' for h in data['hours']))
        self.assertEqual(data['hours'][-1]['status'], 'failed')
        self.assertFalse(data['hours'][-1]['observed'])

    def test_cleanup_archives_unarchived_run_before_deletion(self):
        from app.services.live_trend_snapshots import _cleanup_old_runs
        from app.core.config import settings
        run = self.run_sample(self.now)
        self.db.query(TrendHistoryBucket).delete()
        self.db.query(TrendHistoryAttempt).delete()
        self.db.commit()
        with patch.object(settings, 'live_trend_snapshot_retention_runs', 0):
            _cleanup_old_runs(self.db, region='TH', snapshot_kind='global')
        self.db.commit()
        self.assertEqual(self.db.query(TrendSnapshotRun).count(), 0)
        self.assertEqual(self.history()['points'][0]['run_id'], run.run_id)

    def test_google_does_not_expose_category_shares_or_add_search_volumes(self):
        self.run_sample(self.now, platform='google')
        data = load_trend_history(self.db, region='TH', platform='google', now=self.now)
        self.assertEqual(data['points'][0]['category_counts'], {})
        self.assertNotIn('search_volume', data['points'][0])

    def test_public_endpoint_and_invalid_parameters(self):
        app = FastAPI()
        app.include_router(dashboard.router)
        app.dependency_overrides[get_db] = lambda: MagicMock()
        with TestClient(app) as client:
            with patch('app.routes.dashboard.load_trend_history', return_value={'points': []}) as load:
                self.assertEqual(client.get('/dashboard/public/history?platform=youtube').status_code, 200)
                load.assert_called_once()
            for query in ['platform=tiktok', 'platform=youtube&days=91',
                          'platform=youtube&region=THH', 'platform=youtube&video_category_id=abc']:
                self.assertEqual(client.get(f'/dashboard/public/history?{query}').status_code, 422)
            self.assertEqual(client.get('/dashboard/public/history?platform=google&video_category_id=20').status_code, 400)
            self.assertEqual(client.get('/dashboard/public/history?platform=youtube&days=2').status_code, 400)


if __name__ == '__main__':
    unittest.main()
