import '../models/recommendation_result.dart';
import '../services/api_client.dart';

class AnalysisRepository {
  AnalysisRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<AnalysisResultViewData> analyzeAndSaveVideo(String filePath) async {
    final response = await _client.postMultipart('/analyze/save', filePath);
    return AnalysisResultViewData.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }
}
