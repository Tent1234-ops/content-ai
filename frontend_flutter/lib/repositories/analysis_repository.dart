import 'dart:async';
import 'dart:typed_data';

import '../models/recommendation_result.dart';
import '../services/api_client.dart';

class AnalysisRepository {
  AnalysisRepository({ApiClient? client}) : _client = client ?? ApiClient();

  final ApiClient _client;

  Future<AnalysisResultViewData> analyzeAndSaveVideo({
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
    return AnalysisResultViewData.fromJson(
      Map<String, dynamic>.from(response as Map),
    );
  }
}
