import 'dart:async';

import 'package:content_ai_web/models/trend_history.dart';
import 'package:content_ai_web/repositories/dashboard_repository.dart';
import 'package:content_ai_web/widgets/trend_history_panel.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> point(String time, Map<String, int> ranks,
        {bool gap = false, int total = 10}) =>
    {
      'observed_at': '2026-09-19T$time:00Z',
      'run_id': 2,
      'break_before': gap,
      'total': total,
      'ranks': ranks,
      'category_counts': {'Gaming': 2},
    };

TrendHistory fixture() => TrendHistory.fromJson({
      'items': [
        {'key': 'a', 'title': 'คลิปตัวอย่าง', 'latest_rank': 2}
      ],
      'points': [
        point('09:00', {'a': 8}),
        point('09:15', {'a': 2})
      ],
    });

class HistoryRepository extends DashboardRepository {
  final requests = <String>[];
  Future<TrendHistory> Function(String)? handler;
  @override
  Future<TrendHistory> getTrendHistory(
      {required String platform, int days = 5, String? categoryId}) async {
    requests.add('$platform:$categoryId:$days');
    return handler == null ? fixture() : await handler!(platform);
  }
}

Widget panel(HistoryRepository repo,
        {String platform = 'youtube',
        String? category,
        String revision = '1'}) =>
    MaterialApp(
        home: Scaffold(
            body: SingleChildScrollView(
      child: TrendHistoryPanel(
          repository: repo,
          platform: platform,
          categoryId: category,
          revision: revision,
          categoryName: (c) => c),
    )));

void main() {
  test(
      'rank gaps and absent video never become zero; best rank is above worse rank',
      () {
    final points = [
      point('08:00', {'a': 8}),
      point('09:00', {}),
      point('11:00', {'a': 2}, gap: true)
    ].map(HistoryPoint.fromJson).toList();
    final chart = HistoryLineChart(
        points: points,
        rank: true,
        color: Colors.blue,
        value: (p) => p.ranks['a']?.toDouble(),
        tooltip: (_) => '');
    expect(chart.spots.where((p) => p == FlSpot.nullSpot).length, 2);
    expect(chart.spots.first.y, -8);
    expect(chart.spots.last.y, -2);
    expect(points.first.share('Gaming'), 20);
    expect(HistoryPoint.fromJson(point('12:00', {}, total: 0)).share('Gaming'),
        isNull);
  });

  testWidgets('public history has working charts, periods and evidence table',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(1440, 1000));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final repo = HistoryRepository();
    await tester.pumpWidget(panel(repo));
    await tester.pumpAndSettle();
    expect(find.byType(LineChart), findsNWidgets(2));
    expect(find.text('หมวดไหนติดอันดับรวมมากขึ้น'), findsOneWidget);
    await tester.tap(find.text('24 ชั่วโมง'));
    await tester.pumpAndSettle();
    expect(repo.requests.last, 'youtube:null:1');
    await tester.tap(find.byTooltip('ดูตารางข้อมูลกราฟ'));
    await tester.pumpAndSettle();
    expect(find.byType(DataTable), findsOneWidget);
    expect(find.text('#8'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('Google and category scope hide global category chart',
      (tester) async {
    final repo = HistoryRepository();
    await tester.pumpWidget(panel(repo, platform: 'google'));
    await tester.pumpAndSettle();
    expect(find.byType(LineChart), findsOneWidget);
    expect(find.text('หมวดไหนติดอันดับรวมมากขึ้น'), findsNothing);
    expect(find.text('ลำดับจาก Google Trends ไม่ใช่อันดับจำนวนผู้ค้นหาทั้งหมด'),
        findsOneWidget);
    await tester.pumpWidget(panel(repo, category: '20'));
    await tester.pumpAndSettle();
    expect(repo.requests.last, 'youtube:20:5');
    expect(find.byType(LineChart), findsOneWidget);
  });

  testWidgets('late old platform response cannot replace current platform',
      (tester) async {
    final old = Completer<TrendHistory>();
    final repo = HistoryRepository()
      ..handler = (platform) => platform == 'youtube'
          ? old.future
          : Future.value(TrendHistory.fromJson({}));
    await tester.pumpWidget(panel(repo));
    await tester.pump();
    await tester.pumpWidget(panel(repo, platform: 'google'));
    await tester.pumpAndSettle();
    old.complete(fixture());
    await tester.pumpAndSettle();
    expect(find.text('ยังไม่มีประวัติที่เก็บได้ในช่วงนี้'), findsOneWidget);
    expect(find.byType(LineChart), findsNothing);
  });

  testWidgets('empty, one point and error retry do not fabricate history',
      (tester) async {
    final repo = HistoryRepository()
      ..handler = (_) => Future.error(Exception('offline'));
    await tester.pumpWidget(panel(repo));
    await tester.pumpAndSettle();
    expect(find.byTooltip('ลองโหลดประวัติอีกครั้ง'), findsOneWidget);
    repo.handler = (_) async => TrendHistory.fromJson({
          'items': [
            {'key': 'a', 'title': 'one'}
          ],
          'points': [
            point('10:00', {'a': 1})
          ]
        });
    await tester.tap(find.byTooltip('ลองโหลดประวัติอีกครั้ง'));
    await tester.pumpAndSettle();
    expect(find.byType(LineChart), findsNothing);
    expect(find.text('ยังมีจุดข้อมูลของรายการนี้ไม่พอเปรียบเทียบ'),
        findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('compact web layout has no overflow and revision reloads',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(850, 950));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final repo = HistoryRepository();
    await tester.pumpWidget(panel(repo));
    await tester.pumpAndSettle();
    await tester.pumpWidget(panel(repo, revision: '2'));
    await tester.pumpAndSettle();
    expect(repo.requests.length, 2);
    expect(find.byType(LineChart), findsNWidgets(2));
    expect(tester.takeException(), isNull);
  });
}
