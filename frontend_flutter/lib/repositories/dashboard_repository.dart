import '../models/dashboard_overview.dart';
import '../services/api_client.dart';

class DashboardRepository {
  DashboardRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<DashboardOverview> getOverview() async {
    final response =
        await _client.get('/dashboard/overview?trend_mode=auto&trend_limit=5');
    return DashboardOverview.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }
}
