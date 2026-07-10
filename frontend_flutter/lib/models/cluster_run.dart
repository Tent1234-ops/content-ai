import 'common_models.dart';

class ClusterRunSummary {
  const ClusterRunSummary({
    required this.runId,
    required this.algorithm,
    required this.nClusters,
    required this.featureDimension,
    required this.membershipCount,
    required this.inertia,
  });

  final int runId;
  final String algorithm;
  final int nClusters;
  final int featureDimension;
  final int membershipCount;
  final num inertia;

  factory ClusterRunSummary.fromJson(Map<String, dynamic> json) {
    return ClusterRunSummary(
      runId: (json['run_id'] as num?)?.toInt() ?? 0,
      algorithm: json['algorithm']?.toString() ?? '-',
      nClusters: (json['n_clusters'] as num?)?.toInt() ?? 0,
      featureDimension: (json['feature_dimension'] as num?)?.toInt() ?? 0,
      membershipCount: (json['membership_count'] as num?)?.toInt() ?? 0,
      inertia: json['inertia'] as num? ?? 0,
    );
  }
}

class ClusterBreakdownItem {
  const ClusterBreakdownItem({
    required this.clusterName,
    required this.memberCount,
  });

  final String clusterName;
  final int memberCount;

  factory ClusterBreakdownItem.fromJson(Map<String, dynamic> json) {
    return ClusterBreakdownItem(
      clusterName: json['cluster_name']?.toString() ?? '-',
      memberCount: (json['member_count'] as num?)?.toInt() ?? 0,
    );
  }

  ChartItem toChartItem() {
    return ChartItem(label: clusterName, count: memberCount);
  }
}

class ClusterMembership {
  const ClusterMembership({
    required this.clusterName,
    required this.topTerms,
    required this.itemText,
  });

  final String clusterName;
  final String topTerms;
  final String itemText;

  factory ClusterMembership.fromJson(Map<String, dynamic> json) {
    return ClusterMembership(
      clusterName: json['cluster_name']?.toString() ?? '-',
      topTerms: json['top_terms']?.toString() ?? '-',
      itemText: json['item_text']?.toString() ?? '',
    );
  }
}

class ClusterRunDetail {
  const ClusterRunDetail({
    required this.runId,
    required this.algorithm,
    required this.nClusters,
    required this.featureDimension,
    required this.membershipCount,
    required this.inertia,
    required this.clusterBreakdown,
    required this.recentMemberships,
  });

  final int runId;
  final String algorithm;
  final int nClusters;
  final int featureDimension;
  final int membershipCount;
  final num inertia;
  final List<ClusterBreakdownItem> clusterBreakdown;
  final List<ClusterMembership> recentMemberships;

  factory ClusterRunDetail.fromJson(Map<String, dynamic> json) {
    return ClusterRunDetail(
      runId: (json['run_id'] as num?)?.toInt() ?? 0,
      algorithm: json['algorithm']?.toString() ?? '-',
      nClusters: (json['n_clusters'] as num?)?.toInt() ?? 0,
      featureDimension: (json['feature_dimension'] as num?)?.toInt() ?? 0,
      membershipCount: (json['membership_count'] as num?)?.toInt() ?? 0,
      inertia: json['inertia'] as num? ?? 0,
      clusterBreakdown:
          (json['cluster_breakdown'] as List<dynamic>? ?? const [])
              .map((item) => ClusterBreakdownItem.fromJson(
                  Map<String, dynamic>.from(item as Map)))
              .toList(),
      recentMemberships:
          (json['recent_memberships'] as List<dynamic>? ?? const [])
              .map((item) => ClusterMembership.fromJson(
                  Map<String, dynamic>.from(item as Map)))
              .toList(),
    );
  }
}

class DatasetClusterRunResult {
  const DatasetClusterRunResult({
    required this.runId,
    required this.totalItemsUsed,
    required this.algorithm,
    required this.nClusters,
  });

  final int? runId;
  final int totalItemsUsed;
  final String algorithm;
  final int nClusters;

  factory DatasetClusterRunResult.fromJson(Map<String, dynamic> json) {
    final result =
        Map<String, dynamic>.from((json['result'] as Map?) ?? const {});
    return DatasetClusterRunResult(
      runId: (json['run_id'] as num?)?.toInt(),
      totalItemsUsed: (json['total_items_used'] as num?)?.toInt() ?? 0,
      algorithm: result['algorithm']?.toString() ?? '-',
      nClusters: (result['n_clusters'] as num?)?.toInt() ?? 0,
    );
  }
}
