import 'common_models.dart';
import 'recommendation_result.dart';

class AdminOverviewReport {
  const AdminOverviewReport({
    required this.datasetTotal,
    required this.clusterRunTotal,
    required this.systemLogTotal,
    required this.topSources,
    required this.topCategories,
    required this.statusBreakdown,
  });

  final int datasetTotal;
  final int clusterRunTotal;
  final int systemLogTotal;
  final List<ChartItem> topSources;
  final List<ChartItem> topCategories;
  final List<ChartItem> statusBreakdown;

  factory AdminOverviewReport.fromJson(Map<String, dynamic> json) {
    return AdminOverviewReport(
      datasetTotal: (json['dataset_total'] as num?)?.toInt() ?? 0,
      clusterRunTotal: (json['cluster_run_total'] as num?)?.toInt() ?? 0,
      systemLogTotal: (json['system_log_total'] as num?)?.toInt() ?? 0,
      topSources: (json['top_sources'] as List<dynamic>? ?? const [])
          .map((item) => ChartItem.fromJson(
              Map<String, dynamic>.from(item as Map),
              labelKey: 'source_platform'))
          .toList(),
      topCategories: (json['top_categories'] as List<dynamic>? ?? const [])
          .map((item) => ChartItem.fromJson(
              Map<String, dynamic>.from(item as Map),
              labelKey: 'category'))
          .toList(),
      statusBreakdown: (json['status_breakdown'] as List<dynamic>? ?? const [])
          .map((item) => ChartItem.fromJson(
              Map<String, dynamic>.from(item as Map),
              labelKey: 'status'))
          .toList(),
    );
  }
}

class DatasetHealth {
  const DatasetHealth({
    required this.totalDatasetContents,
    required this.durationCoverageCount,
    required this.durationCoverageRatio,
  });

  final int totalDatasetContents;
  final int durationCoverageCount;
  final double durationCoverageRatio;

  factory DatasetHealth.fromJson(Map<String, dynamic> json) {
    return DatasetHealth(
      totalDatasetContents:
          (json['total_dataset_contents'] as num?)?.toInt() ?? 0,
      durationCoverageCount:
          (json['duration_coverage_count'] as num?)?.toInt() ?? 0,
      durationCoverageRatio:
          (json['duration_coverage_ratio'] as num?)?.toDouble() ?? 0,
    );
  }
}

class ProfileHealth {
  const ProfileHealth({
    required this.youtubeProfiles,
    required this.googleProfiles,
    required this.youtubeDomains,
    required this.googleDomains,
  });

  final int youtubeProfiles;
  final int googleProfiles;
  final List<String> youtubeDomains;
  final List<String> googleDomains;

  factory ProfileHealth.fromJson(Map<String, dynamic> json) {
    return ProfileHealth(
      youtubeProfiles: (json['youtube_profiles'] as num?)?.toInt() ?? 0,
      googleProfiles: (json['google_profiles'] as num?)?.toInt() ?? 0,
      youtubeDomains: (json['youtube_domains'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      googleDomains: (json['google_domains'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class DatasetProfile {
  const DatasetProfile({
    required this.domain,
    required this.sampleSize,
    required this.duration,
  });

  final String domain;
  final int sampleSize;
  final DurationRecommendation duration;

  factory DatasetProfile.fromJson(Map<String, dynamic> json) {
    return DatasetProfile(
      domain: json['domain']?.toString() ?? '-',
      sampleSize: (json['sample_size'] as num?)?.toInt() ?? 0,
      duration: DurationRecommendation.fromJson(
        Map<String, dynamic>.from(
            (json['recommended_duration'] as Map?) ?? const {}),
      ),
    );
  }
}

class RecommendationAdminReport {
  const RecommendationAdminReport({
    required this.datasetHealth,
    required this.profileHealth,
    required this.youtubeProfiles,
    required this.googleProfiles,
  });

  final DatasetHealth datasetHealth;
  final ProfileHealth profileHealth;
  final List<DatasetProfile> youtubeProfiles;
  final List<DatasetProfile> googleProfiles;

  factory RecommendationAdminReport.fromJson(Map<String, dynamic> json) {
    return RecommendationAdminReport(
      datasetHealth: DatasetHealth.fromJson(
        Map<String, dynamic>.from((json['dataset_health'] as Map?) ?? const {}),
      ),
      profileHealth: ProfileHealth.fromJson(
        Map<String, dynamic>.from((json['profile_health'] as Map?) ?? const {}),
      ),
      youtubeProfiles: (json['youtube_profiles'] as List<dynamic>? ?? const [])
          .map((item) =>
              DatasetProfile.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      googleProfiles: (json['google_profiles'] as List<dynamic>? ?? const [])
          .map((item) =>
              DatasetProfile.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }
}

class ProfileComparison {
  const ProfileComparison({
    required this.domain,
    required this.leftSampleSize,
    required this.rightSampleSize,
    required this.leftDuration,
    required this.rightDuration,
    required this.leftTopKeywords,
    required this.rightTopKeywords,
  });

  final String domain;
  final int leftSampleSize;
  final int rightSampleSize;
  final DurationRecommendation leftDuration;
  final DurationRecommendation rightDuration;
  final List<KeywordScore> leftTopKeywords;
  final List<KeywordScore> rightTopKeywords;

  factory ProfileComparison.fromJson(Map<String, dynamic> json) {
    return ProfileComparison(
      domain: json['domain']?.toString() ?? '-',
      leftSampleSize: (json['left_sample_size'] as num?)?.toInt() ?? 0,
      rightSampleSize: (json['right_sample_size'] as num?)?.toInt() ?? 0,
      leftDuration: DurationRecommendation.fromJson(
        Map<String, dynamic>.from((json['left_duration'] as Map?) ?? const {}),
      ),
      rightDuration: DurationRecommendation.fromJson(
        Map<String, dynamic>.from((json['right_duration'] as Map?) ?? const {}),
      ),
      leftTopKeywords: (json['left_top_keywords'] as List<dynamic>? ?? const [])
          .map((item) =>
              KeywordScore.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      rightTopKeywords:
          (json['right_top_keywords'] as List<dynamic>? ?? const [])
              .map((item) =>
                  KeywordScore.fromJson(Map<String, dynamic>.from(item as Map)))
              .toList(),
    );
  }
}
