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
  });

  final String title;
  final String sourcePlatform;
  final num trendScore;

  factory DashboardTrendItem.fromJson(Map<String, dynamic> json) {
    return DashboardTrendItem(
      title: json['title']?.toString() ?? '-',
      sourcePlatform: (json['source_platform'] as String?) ?? json['source']?.toString() ?? '-',
      trendScore: json['trend_score'] as num? ?? 0,
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
    required this.liveYoutubeTrends,
    required this.liveYoutubeTrendMode,
    required this.sourceDistribution,
  });

  final String userRole;
  final DashboardMetrics metrics;
  final List<PlatformComparisonItem> platformComparison;
  final List<PlatformSummary> platformSummaries;
  final List<DashboardTrendItem> topTrends;
  final List<DashboardTrendItem> liveYoutubeTrends;
  final String liveYoutubeTrendMode;
  final List<ChartItem> sourceDistribution;

  bool get isAdmin => userRole == 'admin';

  factory DashboardOverview.fromJson(Map<String, dynamic> json) {
    return DashboardOverview(
      userRole: json['user_role']?.toString() ?? 'user',
      metrics: DashboardMetrics.fromJson(
        Map<String, dynamic>.from((json['metrics'] as Map?) ?? const {}),
      ),
      platformComparison:
          (json['platform_comparison'] as List<dynamic>? ?? const [])
              .map((item) => PlatformComparisonItem.fromJson(
                  Map<String, dynamic>.from(item as Map)))
              .toList(),
      platformSummaries: (json['platform_summaries'] as List<dynamic>? ??
              const [])
          .map((item) =>
              PlatformSummary.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      topTrends: (json['top_trends'] as List<dynamic>? ?? const [])
          .map((item) => DashboardTrendItem.fromJson(
              Map<String, dynamic>.from(item as Map)))
          .toList(),
      liveYoutubeTrends: (json['youtube_trends'] != null
              ? (json['youtube_trends']['items'] as List<dynamic>?)
              : null)
          ?.map((item) => DashboardTrendItem.fromJson(
              Map<String, dynamic>.from(item as Map)))
          .toList() ??
              const [],
      liveYoutubeTrendMode: json['youtube_trends'] is Map
          ? json['youtube_trends']['mode']?.toString() ?? 'unknown'
          : 'unknown',
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
