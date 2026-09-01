import 'package:content_ai_web/models/dashboard_overview.dart';
import 'package:content_ai_web/repositories/dashboard_repository.dart';
import 'package:content_ai_web/screens/dashboard_screen.dart';
import 'package:content_ai_web/state/auth_controller.dart';
import 'package:content_ai_web/state/auth_scope.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('dashboard keeps platform rankings in separate tabs',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byType(TabBar), findsOneWidget);
    expect(find.text('YouTube'), findsOneWidget);
    expect(find.text('Google'), findsOneWidget);
    expect(find.text('TikTok'), findsOneWidget);
    expect(find.text('Current Trend Strength'), findsNothing);
    expect(find.text('Fastest Rising'), findsNothing);
    expect(find.text('Stable'), findsNothing);
    expect(find.text('Momentum Score'), findsNothing);

    await tester.scrollUntilVisible(
      find.text('อันดับวิดีโอบน YouTube ตอนนี้').first,
      250,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('อันดับวิดีโอบน YouTube ตอนนี้'), findsOneWidget);
    expect(find.text('อันดับคำค้นบน Google ตอนนี้'), findsNothing);

    await tester.tap(find.text('Google').first);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('อันดับคำค้นบน Google ตอนนี้').first,
      -250,
      scrollable: find.byType(Scrollable).first,
    );

    expect(find.text('อันดับคำค้นบน Google ตอนนี้'), findsOneWidget);
    expect(find.text('อันดับวิดีโอบน YouTube ตอนนี้'), findsNothing);
    expect(find.text('Google search trend'), findsOneWidget);
    expect(find.text('YouTube video 1'), findsNothing);

    const firstRoundMessage =
        'ยังไม่มี Snapshot รอบก่อนสำหรับเปรียบเทียบอันดับ';
    await tester.drag(
      find.byType(Scrollable).first,
      const Offset(0, -500),
    );
    await tester.pumpAndSettle();
    expect(find.text(firstRoundMessage), findsOneWidget);
  });

  testWidgets('rank movement list explains the exact snapshot comparison',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();

    await tester.scrollUntilVisible(
      find.text('อันดับขยับขึ้นล่าสุด').first,
      500,
      maxScrolls: 20,
      scrollable: find.byType(Scrollable).first,
    );

    expect(find.text('อันดับขยับขึ้นล่าสุด'), findsOneWidget);
    expect(find.text('น่าจับตาในรอบล่าสุด'), findsNothing);
    expect(find.text('YouTube climber'), findsWidgets);
    expect(
      find.text('ขยับขึ้น 5 อันดับ · #18 → #13'),
      findsOneWidget,
    );
    expect(find.text('เข้าอันดับรอบนี้ที่ #14'), findsOneWidget);
    expect(find.textContaining('เปรียบเทียบอันดับรอบ'), findsOneWidget);
    expect(find.textContaining('ภายใน YouTube เท่านั้น'), findsOneWidget);
    expect(find.textContaining('Momentum'), findsNothing);
    expect(find.textContaining('ความสนใจเพิ่มขึ้น'), findsNothing);
  });

  testWidgets('youtube category uses its own 1 to 50 ranking', (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();

    final categoryDropdown = find.byType(DropdownButtonFormField<String>);
    await tester.scrollUntilVisible(
      categoryDropdown,
      250,
      scrollable: find.byType(Scrollable).first,
    );
    await tester.tap(categoryDropdown);
    await tester.pumpAndSettle();
    await tester.tap(find.text('บันเทิง (50)').last);
    await tester.pumpAndSettle();

    expect(find.text('อันดับวิดีโอ YouTube หมวดบันเทิง'), findsOneWidget);
    expect(find.text('Category 24 video 1'), findsOneWidget);
    expect(find.byKey(const Key('top-trend-scroll')), findsOneWidget);
    await tester.scrollUntilVisible(
      find.text('Category 24 video 50'),
      500,
      maxScrolls: 20,
      scrollable: find.descendant(
        of: find.byKey(const Key('top-trend-scroll')),
        matching: find.byType(Scrollable),
      ),
    );
    expect(find.text('Category 24 video 50'), findsOneWidget);
    expect(find.text('#50'), findsOneWidget);
    expect(find.text('YouTube video 1'), findsNothing);
  });

  testWidgets('top trend opens details from the current list without history',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('YouTube video 1'),
      250,
      scrollable: find.byType(Scrollable).first,
    );

    await tester.tap(find.text('YouTube video 1'));
    await tester.pumpAndSettle();

    expect(find.text('รายละเอียดเทรนด์'), findsOneWidget);
    expect(find.text('Test creator channel'), findsOneWidget);
    expect(find.text('1,234'), findsOneWidget);
    expect(find.text('2:05'), findsOneWidget);
    expect(find.text('คำอธิบายจากช่อง'), findsOneWidget);
    expect(find.text('Description supplied by the channel'), findsOneWidget);
    expect(find.text('ดูบน YouTube'), findsOneWidget);
    expect(find.text('ประวัติอันดับที่ระบบเก็บไว้'), findsNothing);
    expect(find.textContaining('โหลดประวัติเพิ่มเติมไม่สำเร็จ'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('trend details hide metadata fields that are unavailable',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('YouTube video 2'),
      250,
      scrollable: find.byType(Scrollable).first,
    );

    await tester.tap(find.text('YouTube video 2'));
    await tester.pumpAndSettle();

    expect(find.text('รายละเอียดเทรนด์'), findsOneWidget);
    expect(find.text('Test creator channel'), findsNothing);
    expect(find.text('ความยาว'), findsNothing);
    expect(find.text('สถิติปัจจุบันจาก YouTube'), findsNothing);
    expect(find.text('คำอธิบายจากช่อง'), findsNothing);
    expect(find.text('ไม่มีข้อมูล'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('google trend details show approximate search-volume evidence',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1280, 900));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.text('Google').first);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('Google search trend'),
      250,
      scrollable: find.byType(Scrollable).first,
    );

    await tester.tap(find.text('Google search trend'));
    await tester.pumpAndSettle();

    expect(find.text('จำนวนการค้นหาโดยประมาณ'), findsOneWidget);
    expect(find.text('ประมาณ 50,000 ครั้ง'), findsOneWidget);
    expect(find.textContaining('ไม่ใช่จำนวนผู้ใช้แบบไม่ซ้ำ'), findsOneWidget);
    expect(
        find.textContaining('อันดับ #1 คือลำดับ Trending Now'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('platform tabs fit a narrow web viewport', (tester) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byType(TabBar), findsOneWidget);
    expect(find.text('YouTube'), findsOneWidget);
    expect(find.text('Google'), findsOneWidget);
    expect(find.text('TikTok'), findsOneWidget);
    expect(tester.takeException(), isNull);

    await tester.tap(find.text('TikTok'));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'trend detail uses a full-width narrow web panel without overflow',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final authController = AuthController();
    addTearDown(authController.dispose);

    await tester.pumpWidget(
      MaterialApp(
        home: AuthScope(
          controller: authController,
          child: DashboardScreen(repository: _FakeDashboardRepository()),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('YouTube video 1'),
      250,
      scrollable: find.byType(Scrollable).first,
    );
    await tester.tap(find.text('YouTube video 1'));
    await tester.pumpAndSettle();

    expect(find.text('รายละเอียดเทรนด์'), findsOneWidget);
    expect(find.text('อันดับปัจจุบัน'), findsOneWidget);
    expect(find.text('ดูบน YouTube'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}

class _FakeDashboardRepository extends DashboardRepository {
  _FakeDashboardRepository()
      : _snapshot = LiveTrendSnapshot.fromJson(_snapshotJson()),
        _overview = DashboardOverview.fromJson(_overviewJson());

  final LiveTrendSnapshot _snapshot;
  final DashboardOverview _overview;

  @override
  Future<DashboardOverview> getOverview() async => _overview;

  @override
  Future<LiveTrendSnapshot> getLiveTrendSnapshot({int limit = 50}) async =>
      _snapshot;

  @override
  Future<YouTubeCategoryTrendSnapshot> getYouTubeCategoryTrendSnapshot({
    String? categoryId,
    int limit = 50,
  }) async =>
      YouTubeCategoryTrendSnapshot.fromJson(
        _youtubeCategorySnapshotJson(categoryId: categoryId),
      );

  @override
  Future<List<FollowedTopicItem>> getFollowedTopics() async => const [];

  @override
  Future<List<NotificationItem>> getNotifications({
    bool unreadOnly = false,
    int limit = 20,
  }) async =>
      const [];
}

Map<String, dynamic> _overviewJson() {
  final snapshot = _snapshotJson();
  final platforms = snapshot['platforms'] as Map<String, dynamic>;
  return {
    'user_role': 'user',
    'metrics': {
      'total_dataset_contents': 0,
      'total_users': 0,
      'my_analysis_results': 0,
    },
    'youtube_trends': platforms['youtube'],
    'google_trends': platforms['google'],
    'tiktok_trends': platforms['tiktok'],
    'top_trends': <dynamic>[],
    'platform_summaries': <dynamic>[],
    'platform_comparison': <dynamic>[],
    'source_distribution': <dynamic>[],
  };
}

Map<String, dynamic> _snapshotJson() {
  final youtubeItems = List.generate(
    12,
    (index) => _trendItem(
      title: 'YouTube video ${index + 1}',
      platform: 'youtube_live',
      rank: index + 1,
      category: 'Technology',
      includeMetadata: index != 1,
    ),
  )
    ..add(
      _trendItem(
        title: 'YouTube climber',
        platform: 'youtube_live',
        rank: 13,
        rankChange: 5,
        category: 'Technology',
        changeKind: 'rank_up',
        meaningfulRising: true,
      ),
    )
    ..add(
      _trendItem(
        title: 'YouTube newcomer',
        platform: 'youtube_live',
        rank: 14,
        category: 'Technology',
        isNew: true,
        hasPreviousSnapshot: false,
      ),
    );
  return {
    'generated_at': '2026-08-31T10:30:00Z',
    'new_count': 0,
    'new_notifications': <dynamic>[],
    'platforms': {
      'youtube': {'mode': 'live', 'items': youtubeItems},
      'google': {
        'mode': 'live',
        'items': [
          _trendItem(
            title: 'Google search trend',
            platform: 'google_trends_live',
            rank: 1,
            category: 'Search',
            changeKind: 'baseline',
            hasPreviousSnapshot: false,
          ),
        ],
      },
      'tiktok': {'mode': 'live', 'items': <dynamic>[]},
    },
  };
}

Map<String, dynamic> _youtubeCategorySnapshotJson({String? categoryId}) {
  final categories = [
    {
      'category_id': '24',
      'title': 'Entertainment',
      'total': 50,
      'provider_status': 'ok',
    },
    {
      'category_id': '20',
      'title': 'Gaming',
      'total': 50,
      'provider_status': 'ok',
    },
  ];
  final selected = categoryId == null
      ? null
      : {
          ...categories.firstWhere(
            (item) => item['category_id'] == categoryId,
          ),
          'ranking_scope': 'category:$categoryId',
          'has_previous_snapshot': true,
          'items': List.generate(
            50,
            (index) => {
              ..._trendItem(
                title: 'Category $categoryId video ${index + 1}',
                platform: 'youtube_live',
                rank: index + 1,
                category: categoryId == '24' ? 'Entertainment' : 'Gaming',
                rankChange: index == 1 ? 3 : 0,
                changeKind: index == 1 ? 'rank_up' : 'none',
                meaningfulRising: index == 1,
              ),
              'category_id': categoryId,
              'ranking_scope': 'category:$categoryId',
            },
          ),
        };
  return {
    'run_id': 20,
    'snapshot_status': 'completed',
    'generated_at': '2026-08-31T10:45:00Z',
    'region': 'TH',
    'refresh_interval_seconds': 900,
    'categories': categories,
    'selected_category': selected,
  };
}

Map<String, dynamic> _trendItem({
  required String title,
  required String platform,
  required int rank,
  required String category,
  int rankChange = 0,
  String changeKind = 'none',
  bool meaningfulRising = false,
  bool hasPreviousSnapshot = true,
  bool isNew = false,
  bool includeMetadata = true,
}) {
  return {
    'key': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'platform': platform.contains('youtube')
        ? 'youtube'
        : platform.contains('google')
            ? 'google'
            : 'tiktok',
    'title': title,
    'source_platform': platform,
    'video_url': 'https://example.test/video',
    if (includeMetadata && !platform.contains('google')) ...{
      'channel_title': 'Test creator channel',
      'description': 'Description supplied by the channel',
      'duration_seconds': 125,
      'published_at': '2026-08-31T10:00:00Z',
      'views': 1234,
      'likes': 120,
      'comments': 15,
      'views_available': true,
      'likes_available': true,
      'comments_available': true,
    },
    if (platform.contains('google')) 'trend_score': 50000,
    'category': category,
    'rank': rank,
    'rank_change': rankChange,
    'change_kind': changeKind,
    'change_label': '',
    'is_meaningful_rising': meaningfulRising,
    'has_previous_snapshot': hasPreviousSnapshot,
    'comparison_window_seconds': hasPreviousSnapshot ? 900 : 0,
    'is_new': isNew,
  };
}
