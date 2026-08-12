import 'package:content_ai_mobile/models/recommendation_result.dart';
import 'package:content_ai_mobile/models/dataset_review.dart';
import 'package:content_ai_mobile/repositories/admin_repository.dart';
import 'package:content_ai_mobile/screens/admin_dataset_review_screen.dart';
import 'package:content_ai_mobile/screens/login_screen.dart';
import 'package:content_ai_mobile/state/auth_controller.dart';
import 'package:content_ai_mobile/state/auth_scope.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('classification result displays hierarchy and Unknown/Other', () {
    final classified = ClassificationResult.fromJson({
      'domain': 'headphone',
      'confidence': 0.84,
      'rule_domain': 'audio',
      'source': 'youtube',
      'taxonomy_version': 'content-taxonomy-v1',
      'taxonomy_leaf_key': 'headphone',
      'category_level_1': 'Technology',
      'category_level_2': 'Electronics',
      'category_level_3': 'Headphone',
      'is_unknown': false,
      'taxonomy_ready': true,
      'candidates': <dynamic>[],
    });
    final unknown = ClassificationResult.fromJson({
      'domain': 'unknown',
      'confidence': 0.0,
      'rule_domain': 'keyboard',
      'source': 'youtube',
      'is_unknown': true,
      'candidates': <dynamic>[],
    });

    expect(
      classified.displayCategory,
      'Technology > Electronics > Headphone',
    );
    expect(unknown.displayCategory, 'Unknown/Other');
  });

  test('recommendation evidence keeps dataset lineage', () {
    final evidence = RecommendationEvidence.fromJson({
      'source': 'youtube_cc_human_verified',
      'dataset_sources': ['youtube_cc'],
      'dataset_versions': ['youtube-cc-th-v1'],
      'source_record_ids': ['video000001', 'video000002'],
      'data_source_label':
          'Human-reviewed YouTube Creative Commons transcripts',
      'dataset_sample_size': 2,
      'eligible_pool_size': 5,
      'source_platform_counts': {'youtube': 2},
      'language_counts': {'th': 1, 'en': 1},
      'verification_status': 'human_verified',
      'license_name': 'YouTube Creative Commons Attribution',
    });

    expect(evidence.datasetSources, ['youtube_cc']);
    expect(evidence.datasetVersions, ['youtube-cc-th-v1']);
    expect(evidence.sourceRecordIds.length, 2);
    expect(evidence.eligiblePoolSize, 5);
    expect(evidence.languageCounts, {'th': 1, 'en': 1});
  });

  test('dataset review candidate keeps review controls and checks', () {
    final candidate = DatasetReviewCandidate.fromJson({
      'collection_run_id': 1,
      'dataset_version': 'youtube-cc-th-v1',
      'source_youtube_id': 'ikPAwWtj2qQ',
      'title': 'Galaxy Z Flip 6 Review',
      'video_url': 'https://www.youtube.com/watch?v=ikPAwWtj2qQ',
      'channel_title': 'Tech Maniac',
      'proposed_leaf_key': 'phone',
      'transcript_language': 'en',
      'caption_type': 'manual',
      'duration_seconds': 58,
      'transcript': 'Galaxy Z Flip phone review transcript',
      'transcript_preview': 'Galaxy Z Flip phone review transcript',
      'evidence_terms': ['phone'],
      'automated_checks': {
        'creative_commons': true,
        'within_five_minutes': true,
        'public_transcript': true,
      },
      'views': 11,
      'likes': 0,
      'comments': 0,
      'review_status': 'pending',
    });

    expect(candidate.reviewStatus, 'pending');
    expect(candidate.proposedLeafKey, 'phone');
    expect(candidate.durationSeconds, 58);
    expect(candidate.allAutomatedChecksPass, isTrue);
  });

  testWidgets('login screen renders expected fields',
      (WidgetTester tester) async {
    final controller = AuthController();
    addTearDown(controller.dispose);
    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: controller,
          child: const LoginScreen(),
        ),
      ),
    );

    expect(find.text('Login'), findsAtLeastNWidgets(1));
    expect(find.text('Email'), findsOneWidget);
    expect(find.text('Password'), findsOneWidget);
    expect(find.text('Create account'), findsOneWidget);
  });

  testWidgets('admin dataset review shows approve and reject controls',
      (WidgetTester tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    await tester.pumpWidget(
      MaterialApp(
        home: AdminDatasetReviewScreen(
          repository: _FakeAdminRepository(),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Galaxy Z Flip 6 Review'), findsOneWidget);
    expect(find.widgetWithText(FilledButton, 'Approve'), findsOneWidget);
    expect(find.widgetWithText(OutlinedButton, 'Reject'), findsOneWidget);
    expect(find.text('Phone 0/30'), findsOneWidget);
  });
}

class _FakeAdminRepository extends AdminRepository {
  @override
  Future<DatasetReviewQueueResult> listDatasetReviewQueue({
    required int limit,
    required int offset,
    String status = 'pending',
    String leafKey = 'all',
    int? collectionRunId,
    String search = '',
  }) async {
    return DatasetReviewQueueResult.fromJson({
      'total': 1,
      'limit': limit,
      'offset': offset,
      'summary': {'total': 1, 'pending': 1, 'approved': 0, 'rejected': 0},
      'runs': [
        {
          'collection_run_id': 1,
          'dataset_version': 'youtube-cc-th-v1',
          'status': 'review_pending',
          'started_at': '2026-08-12T12:00:00Z',
          'total': 1,
          'pending': 1,
          'approved': 0,
          'rejected': 0,
        }
      ],
      'taxonomy': [
        {
          'leaf_key': 'phone',
          'category_level_1': 'Technology',
          'category_level_2': 'Electronics',
          'category_level_3': 'Phone',
          'minimum_sample_count': 30,
          'verified_sample_count': 0,
          'ready': false,
        }
      ],
      'items': [
        {
          'collection_run_id': 1,
          'dataset_version': 'youtube-cc-th-v1',
          'source_youtube_id': 'ikPAwWtj2qQ',
          'title': 'Galaxy Z Flip 6 Review',
          'video_url': 'https://www.youtube.com/watch?v=ikPAwWtj2qQ',
          'channel_title': 'Tech Maniac',
          'proposed_leaf_key': 'phone',
          'transcript_language': 'en',
          'caption_type': 'manual',
          'duration_seconds': 58,
          'transcript': 'Galaxy Z Flip phone review transcript',
          'transcript_preview': 'Galaxy Z Flip phone review transcript',
          'evidence_terms': ['phone'],
          'automated_checks': {
            'creative_commons': true,
            'within_five_minutes': true,
            'public_transcript': true,
          },
          'views': 11,
          'likes': 0,
          'comments': 0,
          'review_status': 'pending',
        }
      ],
    });
  }
}
