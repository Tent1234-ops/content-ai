import 'common_models.dart';

class DashboardMetrics {
  const DashboardMetrics({
    required this.totalDatasetContents,
    required this.totalUsers,
    required this.totalClusterRuns,
    required this.myAnalysisResults,
  });

  final int totalDatasetContents;
  final int totalUsers;
  final int totalClusterRuns;
  final int myAnalysisResults;

  factory DashboardMetrics.fromJson(Map<String, dynamic> json) {
    return DashboardMetrics(
      totalDatasetContents:
          (json['total_dataset_contents'] as num?)?.toInt() ?? 0,
      totalUsers: (json['total_users'] as num?)?.toInt() ?? 0,
      totalClusterRuns: (json['total_cluster_runs'] as num?)?.toInt() ?? 0,
      myAnalysisResults: (json['my_analysis_results'] as num?)?.toInt() ?? 0,
    );
  }
}

class PlatformSummary {
  const PlatformSummary({
    required this.source,
    required this.datasetCount,
    required this.profileCount,
    required this.domains,
  });

  final String source;
  final int datasetCount;
  final int profileCount;
  final List<String> domains;

  factory PlatformSummary.fromJson(Map<String, dynamic> json) {
    return PlatformSummary(
      source: json['source']?.toString() ?? '-',
      datasetCount: (json['dataset_count'] as num?)?.toInt() ?? 0,
      profileCount: (json['profile_count'] as num?)?.toInt() ?? 0,
      domains: (json['domains'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class PlatformComparisonItem {
  const PlatformComparisonItem({
    required this.domain,
    required this.youtubeSampleSize,
    required this.googleSampleSize,
    required this.youtubeDuration,
    required this.googleDuration,
  });

  final String domain;
  final int youtubeSampleSize;
  final int googleSampleSize;
  final String youtubeDuration;
  final String googleDuration;

  factory PlatformComparisonItem.fromJson(Map<String, dynamic> json) {
    return PlatformComparisonItem(
      domain: json['domain']?.toString() ?? '-',
      youtubeSampleSize: (json['youtube_sample_size'] as num?)?.toInt() ?? 0,
      googleSampleSize: (json['google_sample_size'] as num?)?.toInt() ?? 0,
      youtubeDuration: json['youtube_duration']?.toString() ?? '-',
      googleDuration: json['google_duration']?.toString() ?? '-',
    );
  }
}

class DashboardTrendItem {
  const DashboardTrendItem({
    required this.title,
    required this.sourcePlatform,
    required this.trendScore,
    required this.category,
    required this.videoUrl,
    required this.views,
    required this.likes,
    required this.comments,
    required this.publishedAt,
  });

  final String title;
  final String sourcePlatform;
  final num trendScore;
  final String category;
  final String videoUrl;
  final int views;
  final int likes;
  final int comments;
  final String publishedAt;

  factory DashboardTrendItem.fromJson(Map<String, dynamic> json) {
    return DashboardTrendItem(
      title: json['title']?.toString() ?? '-',
      sourcePlatform:
          json['source_platform']?.toString() ?? json['source']?.toString() ?? '-',
      trendScore: json['trend_score'] as num? ?? 0,
      category: json['category']?.toString() ?? '',
      videoUrl: json['video_url']?.toString() ?? '',
      views: (json['views'] as num?)?.toInt() ?? 0,
      likes: (json['likes'] as num?)?.toInt() ?? 0,
      comments: (json['comments'] as num?)?.toInt() ?? 0,
      publishedAt: json['published_at']?.toString() ?? '',
    );
  }
}

class DashboardPlatformTrends {
  const DashboardPlatformTrends({
    required this.platform,
    required this.mode,
    required this.items,
  });

  final String platform;
  final String mode;
  final List<DashboardTrendItem> items;

  bool get isLive => mode == 'live';

  factory DashboardPlatformTrends.fromJson(
    String platform,
    Map<String, dynamic>? json,
  ) {
    return DashboardPlatformTrends(
      platform: platform,
      mode: json?['mode']?.toString() ?? 'unknown',
      items: (json?['items'] as List<dynamic>? ?? const [])
          .map((item) => DashboardTrendItem.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
    );
  }
}

class FollowedTopicItem {
  const FollowedTopicItem({
    required this.id,
    required this.matchType,
    required this.value,
    required this.createdAt,
  });

  final int id;
  final String matchType;
  final String value;
  final String createdAt;

  factory FollowedTopicItem.fromJson(Map<String, dynamic> json) {
    return FollowedTopicItem(
      id: (json['id'] as num?)?.toInt() ?? 0,
      matchType: json['match_type']?.toString() ?? 'keyword',
      value: json['value']?.toString() ?? '',
      createdAt: json['created_at']?.toString() ?? '',
    );
  }
}

class NotificationItem {
  const NotificationItem({
    required this.id,
    required this.title,
    required this.message,
    required this.topic,
    required this.sourcePlatform,
    required this.trendScore,
    required this.isRead,
    required this.createdAt,
  });

  final int id;
  final String title;
  final String message;
  final String topic;
  final String sourcePlatform;
  final num trendScore;
  final bool isRead;
  final String createdAt;

  factory NotificationItem.fromJson(Map<String, dynamic> json) {
    return NotificationItem(
      id: (json['notification_id'] as num?)?.toInt() ?? 0,
      title: json['title']?.toString() ?? 'Notification',
      message: json['message']?.toString() ?? json['body']?.toString() ?? '',
      topic: json['topic']?.toString() ?? 'general',
      sourcePlatform: json['source_platform']?.toString() ?? 'system',
      trendScore: json['trend_score'] as num? ?? 0,
      isRead: json['is_read'] == true,
      createdAt: json['created_at']?.toString() ?? '',
    );
  }
}

class TrendSyncResult {
  const TrendSyncResult({
    required this.platform,
    required this.created,
    required this.updated,
    required this.notifications,
    required this.mode,
    required this.totalFetched,
    this.error,
  });

  final String platform;
  final int created;
  final int updated;
  final int notifications;
  final String mode;
  final int totalFetched;
  final String? error;

  bool get failed => error != null;

  factory TrendSyncResult.fromJson(String platform, Map<String, dynamic> json) {
    return TrendSyncResult(
      platform: platform,
      created: (json['created'] as num?)?.toInt() ?? 0,
      updated: (json['updated'] as num?)?.toInt() ?? 0,
      notifications: (json['notifications'] as num?)?.toInt() ?? 0,
      mode: json['mode']?.toString() ?? 'unknown',
      totalFetched: (json['total_fetched'] as num?)?.toInt() ?? 0,
    );
  }

  factory TrendSyncResult.failed(String platform, Object error) {
    return TrendSyncResult(
      platform: platform,
      created: 0,
      updated: 0,
      notifications: 0,
      mode: 'failed',
      totalFetched: 0,
      error: error.toString(),
    );
  }
}

class DashboardOverview {
  const DashboardOverview({
    required this.userRole,
    required this.metrics,
    required this.platformComparison,
    required this.platformSummaries,
    required this.topTrends,
    required this.youtubeTrends,
    required this.googleTrends,
    required this.tiktokTrends,
    required this.sourceDistribution,
  });

  final String userRole;
  final DashboardMetrics metrics;
  final List<PlatformComparisonItem> platformComparison;
  final List<PlatformSummary> platformSummaries;
  final List<DashboardTrendItem> topTrends;
  final DashboardPlatformTrends youtubeTrends;
  final DashboardPlatformTrends googleTrends;
  final DashboardPlatformTrends tiktokTrends;
  final List<ChartItem> sourceDistribution;

  bool get isAdmin => userRole == 'admin';

  List<DashboardPlatformTrends> get platformTrends => [
        youtubeTrends,
        googleTrends,
        tiktokTrends,
      ];

  List<DashboardTrendItem> get liveYoutubeTrends => youtubeTrends.items;
  String get liveYoutubeTrendMode => youtubeTrends.mode;

  factory DashboardOverview.fromJson(Map<String, dynamic> json) {
    return DashboardOverview(
      userRole: json['user_role']?.toString() ?? 'user',
      metrics: DashboardMetrics.fromJson(
        Map<String, dynamic>.from((json['metrics'] as Map?) ?? const {}),
      ),
      platformComparison:
          (json['platform_comparison'] as List<dynamic>? ?? const [])
              .map((item) => PlatformComparisonItem.fromJson(
                    Map<String, dynamic>.from(item as Map),
                  ))
              .toList(),
      platformSummaries: (json['platform_summaries'] as List<dynamic>? ??
              const [])
          .map((item) => PlatformSummary.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
      topTrends: (json['top_trends'] as List<dynamic>? ?? const [])
          .map((item) => DashboardTrendItem.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
      youtubeTrends: DashboardPlatformTrends.fromJson(
        'youtube',
        json['youtube_trends'] is Map
            ? Map<String, dynamic>.from(json['youtube_trends'] as Map)
            : null,
      ),
      googleTrends: DashboardPlatformTrends.fromJson(
        'google',
        json['google_trends'] is Map
            ? Map<String, dynamic>.from(json['google_trends'] as Map)
            : null,
      ),
      tiktokTrends: DashboardPlatformTrends.fromJson(
        'tiktok',
        json['tiktok_trends'] is Map
            ? Map<String, dynamic>.from(json['tiktok_trends'] as Map)
            : null,
      ),
      sourceDistribution:
          (json['source_distribution'] as List<dynamic>? ?? const [])
              .map((item) => ChartItem.fromJson(
                    Map<String, dynamic>.from(item as Map),
                    labelKey: 'source_platform',
                  ))
              .toList(),
    );
  }
}
