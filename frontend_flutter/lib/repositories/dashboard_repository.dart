import '../models/dashboard_overview.dart';
import '../services/api_client.dart';

class DashboardRepository {
  DashboardRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<DashboardOverview> getOverview() async {
    final response =
        await _client.get('/dashboard/summary?trend_mode=live&trend_limit=50');
    return DashboardOverview.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  Future<LiveTrendSnapshot> getLiveTrendSnapshot({int limit = 50}) async {
    final response =
        await _client.get('/dashboard/live-trends/snapshot?trend_limit=$limit');
    return LiveTrendSnapshot.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  Future<List<FollowedTopicItem>> getFollowedTopics() async {
    final response = await _client.get('/follows/topics?limit=100');
    final items = (response['items'] as List<dynamic>? ?? const [])
        .map((item) => FollowedTopicItem.fromJson(
              Map<String, dynamic>.from(item as Map),
            ))
        .toList();
    return items;
  }

  Future<FollowedTopicItem> followTopic(
    String value, {
    String matchType = 'keyword',
  }) async {
    final response = await _client.post('/follows/topic', {
      'match_type': matchType,
      'value': value,
    });
    return FollowedTopicItem.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  Future<void> unfollowTopic(int id) async {
    await _client.delete('/follows/topic/$id');
  }

  Future<List<NotificationItem>> getNotifications({
    bool unreadOnly = false,
    int limit = 20,
  }) async {
    final response = await _client.get(
      '/notifications/?unread_only=$unreadOnly&limit=$limit',
    );
    return (response['items'] as List<dynamic>? ?? const [])
        .map((item) => NotificationItem.fromJson(
              Map<String, dynamic>.from(item as Map),
            ))
        .toList();
  }

  Future<void> markNotificationsRead(List<int> ids) async {
    await _client.post('/notifications/mark_read', {'ids': ids});
  }

  Future<List<TrendSyncResult>> syncAllTrendsLive({int limit = 50}) async {
    final platforms = ['youtube', 'google', 'tiktok'];
    final results = <TrendSyncResult>[];
    for (final platform in platforms) {
      try {
        final response = await _client.post(
          '/trends/$platform/sync?mode=live&limit=$limit',
          <String, dynamic>{},
        );
        results.add(
          TrendSyncResult.fromJson(
            platform,
            Map<String, dynamic>.from(response as Map),
          ),
        );
      } catch (error) {
        results.add(TrendSyncResult.failed(platform, error));
      }
    }
    return results;
  }
}
