import 'package:content_ai_web/models/admin_report.dart';
import 'package:content_ai_web/models/common_models.dart';
import 'package:content_ai_web/models/dataset_item.dart';
import 'package:content_ai_web/models/dataset_review.dart';
import 'package:content_ai_web/models/system_log.dart';
import 'package:content_ai_web/repositories/admin_repository.dart';
import 'package:content_ai_web/screens/admin_console_screen.dart';
import 'package:content_ai_web/screens/admin_datasets_screen.dart';
import 'package:content_ai_web/screens/admin_logs_screen.dart';
import 'package:content_ai_web/utils/system_log_presenter.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('system log parses backend audit fields and exposes Thai labels', () {
    final item = SystemLogItem.fromJson({
      'log_id': 9,
      'action': 'dataset_review_reject',
      'status': 'failed',
      'detail': 'youtube_id=abc',
      'timestamp': '2026-09-03T08:30:00',
      'user_id': 2,
    });

    expect(item.timestamp?.isUtc, isTrue);
    expect(item.userId, 2);
    expect(
      systemLogActionLabel(item.action),
      'ปฏิเสธข้อมูลฝึกก่อนนำเข้าฐานข้อมูล',
    );
    expect(systemLogStatusLabel(item.status), 'ไม่สำเร็จ');
    expect(
      systemLogActionLabel('admin_dataset_training_content_corrected'),
      'แก้ไข Transcript หรือหมวดข้อมูลฝึกที่อนุมัติแล้ว',
    );
  });

  testWidgets('admin console shows only settings backed by current behavior',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(
      MaterialApp(
        home: AdminConsoleScreen(repository: _ConsoleRepository()),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('ข้อมูลพร้อมใช้กับโมเดล'), findsOneWidget);
    expect(find.text('ช่วงเปิดคลิปที่ใช้วิเคราะห์ (วินาที)'), findsOneWidget);
    expect(find.text('Max keywords'), findsNothing);
    expect(find.text('Scan interval hr'), findsNothing);
    expect(find.text('Google Rows'), findsNothing);
    expect(find.text('TikTok Rows'), findsNothing);
    expect(find.text('Visual Summary'), findsNothing);
    expect(find.text('YouTube / Google / TikTok Comparison'), findsNothing);
  });

  testWidgets('system logs present audit events in Thai with time and actor',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    await tester.pumpWidget(
      MaterialApp(
        home: AdminLogsScreen(repository: _LogsRepository()),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('อนุมัติข้อมูลฝึกเข้าสู่ฐานข้อมูล'), findsOneWidget);
    expect(find.text('dataset_review_approve'), findsOneWidget);
    expect(find.text('สำเร็จ'), findsAtLeastNWidgets(1));
    expect(find.text('ผู้ใช้หมายเลข 2'), findsOneWidget);
  });
  testWidgets(
      'dataset editor submits transcript with the canonical taxonomy leaf',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final repository = _DatasetsRepository();

    await tester.pumpWidget(
      MaterialApp(home: AdminDatasetsScreen(repository: repository)),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('Phone review'));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byKey(const ValueKey('dataset-transcript')),
      List.filled(
        4,
        'camera sensor lens aperture photography image quality ',
      ).join(),
    );
    await tester.tap(find.byKey(const ValueKey('dataset-taxonomy-leaf')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Technology > Electronics > Camera').last);
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('dataset-save')));
    await tester.pumpAndSettle();

    expect(repository.updatedDatasetId, 42);
    expect(repository.updatedPayload?['taxonomy_leaf_key'], 'camera');
    expect(
      repository.updatedPayload?['transcript'],
      contains('camera sensor lens'),
    );
    expect(repository.updatedPayload?.containsKey('category'), isFalse);
    expect(
        repository.updatedPayload?.containsKey('transcript_sha256'), isFalse);
  });
}

class _ConsoleRepository extends AdminRepository {
  @override
  Future<RecommendationAdminReport> getRecommendationReport() async {
    return RecommendationAdminReport.fromJson({
      'dataset_health': {
        'total_dataset_contents': 150,
        'youtube_dataset_contents': 150,
        'google_dataset_contents': 0,
        'tiktok_dataset_contents': 0,
        'duration_coverage_count': 45,
        'duration_coverage_ratio': 0.3,
      },
      'profile_health': {
        'youtube_profiles': 3,
        'google_profiles': 0,
        'tiktok_profiles': 0,
        'youtube_domains': ['phone', 'camera', 'laptop'],
        'google_domains': <String>[],
        'tiktok_domains': <String>[],
      },
      'youtube_profiles': <dynamic>[],
      'google_profiles': <dynamic>[],
      'tiktok_profiles': <dynamic>[],
    });
  }

  @override
  Future<AdminSettings> getSettings() async {
    return AdminSettings.fromJson({'hook_analysis_duration': 60});
  }

  @override
  Future<AdminSettings> updateSettings(Map<String, dynamic> payload) async {
    return AdminSettings.fromJson(payload);
  }
}

class _LogsRepository extends AdminRepository {
  @override
  Future<PaginatedResult<SystemLogItem>> listLogs({
    required int limit,
    required int offset,
    String status = 'all',
    String action = '',
  }) async {
    return PaginatedResult(
      total: 1,
      items: [
        SystemLogItem.fromJson({
          'log_id': 1,
          'action': 'dataset_review_approve',
          'status': 'success',
          'detail': 'youtube_id=abc, dataset_id=42',
          'timestamp': '2026-09-03T08:30:00',
          'user_id': 2,
        }),
      ],
    );
  }
}

class _DatasetsRepository extends AdminRepository {
  int? updatedDatasetId;
  Map<String, dynamic>? updatedPayload;

  final DatasetItem item = DatasetItem.fromJson({
    'dataset_id': 42,
    'title': 'Phone review',
    'video_url': 'https://youtu.be/phone000001',
    'transcript': List.filled(
      4,
      'phone battery display camera performance ',
    ).join(),
    'source_platform': 'youtube',
    'category': 'phone',
    'taxonomy_version': 'content-taxonomy-v1',
    'taxonomy_leaf_key': 'phone',
    'category_level_1': 'Technology',
    'category_level_2': 'Electronics',
    'category_level_3': 'Phone',
    'transcript_sha256': List.filled(64, 'a').join(),
    'data_split': 'train',
    'is_training_eligible': true,
    'views': 100,
    'likes': 10,
    'comments': 2,
    'trend_score': 1.0,
    'duration_seconds': 180,
  });

  @override
  Future<List<DatasetReviewTaxonomyLeaf>> listTaxonomyLeaves() async {
    return [
      DatasetReviewTaxonomyLeaf.fromJson({
        'leaf_key': 'phone',
        'category_level_1': 'Technology',
        'category_level_2': 'Electronics',
        'category_level_3': 'Phone',
      }),
      DatasetReviewTaxonomyLeaf.fromJson({
        'leaf_key': 'camera',
        'category_level_1': 'Technology',
        'category_level_2': 'Electronics',
        'category_level_3': 'Camera',
      }),
    ];
  }

  @override
  Future<PaginatedResult<DatasetItem>> listDatasets({
    required int limit,
    required int offset,
    String source = 'all',
    String category = 'all',
    String search = '',
  }) async {
    return PaginatedResult(total: 1, items: [item]);
  }

  @override
  Future<DatasetItem> updateDataset(
    int datasetId,
    Map<String, dynamic> payload,
  ) async {
    updatedDatasetId = datasetId;
    updatedPayload = Map<String, dynamic>.from(payload);
    return item;
  }
}
