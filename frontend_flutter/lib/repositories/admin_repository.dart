import '../models/admin_report.dart';
import '../models/cluster_run.dart';
import '../models/common_models.dart';
import '../models/dataset_item.dart';
import '../models/system_log.dart';
import '../services/api_client.dart';

class AdminRepository {
  AdminRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<PaginatedResult<DatasetItem>> listDatasets({
    required int limit,
    required int offset,
    String source = 'all',
    String category = 'all',
    String search = '',
  }) async {
    final buffer = StringBuffer('/admin/datasets?limit=$limit&offset=$offset');
    if (source != 'all') {
      buffer.write('&source=$source');
    }
    if (category != 'all') {
      buffer.write('&category=${Uri.encodeComponent(category)}');
    }
    if (search.trim().isNotEmpty) {
      buffer.write('&search=${Uri.encodeComponent(search.trim())}');
    }
    final response =
        Map<String, dynamic>.from(await _client.get(buffer.toString()) as Map);
    return PaginatedResult<DatasetItem>(
      total: (response['total'] as num?)?.toInt() ?? 0,
      items: (response['items'] as List<dynamic>? ?? const [])
          .map((item) =>
              DatasetItem.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }

  Future<PaginatedResult<ClusterRunSummary>> listClusterRuns({
    required int limit,
    required int offset,
    String algorithm = 'all',
  }) async {
    final buffer =
        StringBuffer('/admin/clusters/runs?limit=$limit&offset=$offset');
    if (algorithm != 'all') {
      buffer.write('&algorithm=$algorithm');
    }
    final response =
        Map<String, dynamic>.from(await _client.get(buffer.toString()) as Map);
    return PaginatedResult<ClusterRunSummary>(
      total: (response['total'] as num?)?.toInt() ?? 0,
      items: (response['items'] as List<dynamic>? ?? const [])
          .map((item) => ClusterRunSummary.fromJson(
              Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }

  Future<ClusterRunDetail> getClusterRun(int runId) async {
    final response = await _client.get('/admin/clusters/runs/$runId');
    return ClusterRunDetail.fromJson(
        Map<String, dynamic>.from(response as Map));
  }

  Future<DatasetClusterRunResult> runClusteringFromDataset(
    Map<String, dynamic> payload,
  ) async {
    final response = await _client.post('/clustering/from-dataset', payload);
    return DatasetClusterRunResult.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  Future<AdminOverviewReport> getOverviewReport() async {
    final response = await _client.get('/admin/reports/overview');
    return AdminOverviewReport.fromJson(
        Map<String, dynamic>.from(response as Map));
  }

  Future<RecommendationAdminReport> getRecommendationReport() async {
    final response = await _client.get('/recommendations/admin/report');
    return RecommendationAdminReport.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }

  Future<List<ProfileComparison>> compareProfiles() async {
    final response = Map<String, dynamic>.from(
      await _client.get(
        '/recommendations/profiles/compare?left_source=youtube&right_source=google&limit=100',
      ) as Map,
    );
    return (response['comparisons'] as List<dynamic>? ?? const [])
        .map((item) =>
            ProfileComparison.fromJson(Map<String, dynamic>.from(item as Map)))
        .toList();
  }

  Future<PaginatedResult<SystemLogItem>> listLogs({
    required int limit,
    required int offset,
    String status = 'all',
    String action = '',
  }) async {
    final buffer = StringBuffer('/admin/logs?limit=$limit&offset=$offset');
    if (status != 'all') {
      buffer.write('&status=$status');
    }
    if (action.trim().isNotEmpty) {
      buffer.write('&action=${Uri.encodeComponent(action.trim())}');
    }
    final response =
        Map<String, dynamic>.from(await _client.get(buffer.toString()) as Map);
    return PaginatedResult<SystemLogItem>(
      total: (response['total'] as num?)?.toInt() ?? 0,
      items: (response['items'] as List<dynamic>? ?? const [])
          .map((item) =>
              SystemLogItem.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }
}
