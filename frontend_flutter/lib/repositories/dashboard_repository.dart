import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/dashboard_overview.dart';
import '../services/api_client.dart';

class DashboardRepository {
  DashboardRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;
  static const _followedTopicsKey = 'followed_topics';

  Future<DashboardOverview> getOverview() async {
    final response =
        await _client.get('/dashboard/summary?trend_mode=auto&trend_limit=20');
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

  /// Save a trend as an idea (stored locally). Stores JSON strings with title, source, and saved_at
  Future<void> saveIdea(String title, String source) async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'saved_ideas';
    final existing = prefs.getStringList(key) ?? [];
    final entry = {
      'title': title,
      'source': source,
      'saved_at': DateTime.now().toIso8601String(),
    };
    existing.add(jsonEncode(entry));
    await prefs.setStringList(key, existing);
  }

  /// Get saved ideas
  Future<List<Map<String, dynamic>>> getSavedIdeas() async {
    final prefs = await SharedPreferences.getInstance();
    final key = 'saved_ideas';
    final existing = prefs.getStringList(key) ?? [];
    return existing.map((s) {
      try {
        return Map<String, dynamic>.from(jsonDecode(s) as Map);
      } catch (_) {
        return {'title': s};
      }
    }).toList();
  }
}
