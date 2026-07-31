import 'dart:async';
import 'dart:typed_data';

import '../models/recommendation_result.dart';
import '../services/api_client.dart';

class AnalysisRepository {
  AnalysisRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<String> startAnalyzeAndSaveVideo({
    String? filePath,
    Uint8List? fileBytes,
    Stream<List<int>>? fileStream,
    int? fileSize,
    required String fileName,
  }) async {
    final response = await _client.postMultipart(
      '/analyze/save',
      fileName: fileName,
      filePath: filePath,
      fileBytes: fileBytes,
      fileStream: fileStream,
      fileSize: fileSize,
    );
    final payload = Map<String, dynamic>.from(response as Map);
    final jobId = payload['job_id']?.toString();
    if (jobId == null || jobId.isEmpty) {
      throw Exception('Backend did not return a job id.');
    }
    return jobId;
  }

  Future<AnalysisJobStatus> getAnalysisJob(String jobId) async {
    final response = await _client.get('/jobs/$jobId');
    return AnalysisJobStatus.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }
}
