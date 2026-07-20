import 'package:shared_preferences/shared_preferences.dart';

import '../models/dashboard_overview.dart';
import '../services/api_client.dart';

class DashboardRepository {
  DashboardRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;
  static const _followedTopicsKey = 'followed_topics';

  Future<DashboardOverview> getOverview() async {
    final response =
        await _client.get('/dashboard/overview?trend_mode=auto&trend_limit=5');
    return DashboardOverview.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  /// Get list of followed topics from local storage
  Future<List<String>> getFollowedTopics() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_followedTopicsKey) ?? [];
  }

  /// Follow a topic (save to local storage)
  Future<void> followTopic(String topic) async {
    final prefs = await SharedPreferences.getInstance();
    final topics = prefs.getStringList(_followedTopicsKey) ?? [];
    if (!topics.contains(topic)) {
      topics.add(topic);
      await prefs.setStringList(_followedTopicsKey, topics);
    }
  }

  /// Unfollow a topic (remove from local storage)
  Future<void> unfollowTopic(String topic) async {
    final prefs = await SharedPreferences.getInstance();
    final topics = prefs.getStringList(_followedTopicsKey) ?? [];
    topics.remove(topic);
    await prefs.setStringList(_followedTopicsKey, topics);
  }

  /// Clear all followed topics
  Future<void> clearFollowedTopics() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_followedTopicsKey);
  }
}
