import 'common_models.dart';

class DurationRecommendation {
  const DurationRecommendation({
    required this.recommendedRange,
    required this.sampleSize,
    required this.source,
  });

  final String recommendedRange;
  final int sampleSize;
  final String source;

  factory DurationRecommendation.fromJson(Map<String, dynamic> json) {
    return DurationRecommendation(
      recommendedRange: json['recommended_range']?.toString() ?? '-',
      sampleSize: (json['sample_size'] as num?)?.toInt() ?? 0,
      source: json['source']?.toString() ?? '-',
    );
  }
}

class ClassificationCandidate {
  const ClassificationCandidate({
    required this.domain,
    required this.score,
    required this.sampleSize,
    required this.matchedTerms,
  });

  final String domain;
  final double score;
  final int sampleSize;
  final List<String> matchedTerms;

  factory ClassificationCandidate.fromJson(Map<String, dynamic> json) {
    return ClassificationCandidate(
      domain: json['domain']?.toString() ?? '-',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      sampleSize: (json['sample_size'] as num?)?.toInt() ?? 0,
      matchedTerms: (json['matched_terms'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class ClassificationResult {
  const ClassificationResult({
    required this.domain,
    required this.confidence,
    required this.ruleDomain,
    required this.source,
    required this.candidates,
  });

  final String domain;
  final double confidence;
  final String ruleDomain;
  final String source;
  final List<ClassificationCandidate> candidates;

  factory ClassificationResult.fromJson(Map<String, dynamic> json) {
    return ClassificationResult(
      domain: json['domain']?.toString() ?? '-',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      ruleDomain: json['rule_domain']?.toString() ?? '-',
      source: json['source']?.toString() ?? 'youtube',
      candidates: (json['candidates'] as List<dynamic>? ?? const [])
          .map((item) => ClassificationCandidate.fromJson(
              Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }
}

class MissingDimension {
  const MissingDimension({
    required this.name,
    required this.score,
    required this.userStatus,
  });

  final String name;
  final double score;
  final String userStatus;

  factory MissingDimension.fromJson(Map<String, dynamic> json) {
    return MissingDimension(
      name: json['name']?.toString() ?? '-',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      userStatus: json['user_status']?.toString() ?? '-',
    );
  }
}

class RecommendationResult {
  const RecommendationResult({
    required this.domain,
    required this.userKeywords,
    required this.contentKeywords,
    required this.hookTerms,
    required this.missingKeywords,
    required this.hookKeywords,
    required this.missingDimensions,
    required this.duration,
    required this.datasetProfile,
    required this.evidence,
    this.classification,
  });

  final String domain;
  final List<String> userKeywords;
  final List<String> contentKeywords;
  final List<String> hookTerms;
  final List<KeywordScore> missingKeywords;
  final List<KeywordScore> hookKeywords;
  final List<MissingDimension> missingDimensions;
  final DurationRecommendation duration;
  final DatasetProfile datasetProfile;
  final RecommendationEvidence evidence;
  final ClassificationResult? classification;

  factory RecommendationResult.fromJson(Map<String, dynamic> json) {
    return RecommendationResult(
      domain: json['domain']?.toString() ?? '-',
      userKeywords: (json['user_keywords'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      contentKeywords: (json['content_keywords'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      hookTerms: (json['hook_terms'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      missingKeywords: (json['missing_keywords'] as List<dynamic>? ?? const [])
          .map((item) =>
              KeywordScore.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      hookKeywords: (json['hook_keywords'] as List<dynamic>? ?? const [])
          .map((item) =>
              KeywordScore.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      missingDimensions: (json['missing_dimensions'] as List<dynamic>? ??
              const [])
          .map((item) =>
              MissingDimension.fromJson(Map<String, dynamic>.from(item as Map)))
          .toList(),
      duration: DurationRecommendation.fromJson(
        Map<String, dynamic>.from(
            (json['recommended_duration'] as Map?) ?? const {}),
      ),
      datasetProfile: DatasetProfile.fromJson(
        Map<String, dynamic>.from(
            (json['dataset_profile'] as Map?) ?? const {}),
      ),
      evidence: RecommendationEvidence.fromJson(
        Map<String, dynamic>.from((json['evidence'] as Map?) ?? const {}),
      ),
      classification: json['classification'] is Map
          ? ClassificationResult.fromJson(
              Map<String, dynamic>.from(json['classification'] as Map),
            )
          : null,
    );
  }
}

class AnalysisResultViewData {
  const AnalysisResultViewData({
    this.contentId,
    required this.title,
    required this.transcript,
    required this.recommendation,
    required this.fallbackDomain,
    required this.saved,
    required this.raw,
  });

  final int? contentId;
  final String title;
  final String transcript;
  final RecommendationResult recommendation;
  final String fallbackDomain;
  final bool saved;
  final Map<String, dynamic> raw;

  factory AnalysisResultViewData.fromJson(Map<String, dynamic> json) {
    final analysisRoot =
        Map<String, dynamic>.from((json['analysis'] as Map?) ?? const {});
    final analysis = Map<String, dynamic>.from(
        (analysisRoot['analysis'] as Map?) ?? const {});
    return AnalysisResultViewData(
      contentId: (json['content_id'] as num?)?.toInt(),
      title: json['title']?.toString() ??
          analysis['title']?.toString() ??
          'Result',
      transcript: json['transcript']?.toString() ??
          analysisRoot['transcript']?.toString() ??
          '',
      recommendation: RecommendationResult.fromJson(
        Map<String, dynamic>.from((json['recommendation'] as Map?) ?? const {}),
      ),
      fallbackDomain: analysis['domain']?.toString() ?? '-',
      saved: json['saved'] == true || json['content_id'] != null,
      raw: json,
    );
  }
}

class DatasetProfile {
  const DatasetProfile({
    required this.domain,
    required this.sampleSize,
    required this.source,
    required this.sourcePlatformCounts,
    required this.exemplarTitles,
  });

  final String domain;
  final int sampleSize;
  final String source;
  final Map<String, int> sourcePlatformCounts;
  final List<String> exemplarTitles;

  factory DatasetProfile.fromJson(Map<String, dynamic> json) {
    final rawCounts = json['source_platform_counts'];
    return DatasetProfile(
      domain: json['domain']?.toString() ?? '-',
      sampleSize: (json['sample_size'] as num?)?.toInt() ?? 0,
      source: json['source']?.toString() ?? '-',
      sourcePlatformCounts: rawCounts is Map
          ? rawCounts.map(
              (key, value) =>
                  MapEntry(key.toString(), (value as num?)?.toInt() ?? 0),
            )
          : const {},
      exemplarTitles: (json['exemplar_titles'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
    );
  }
}

class RecommendationEvidence {
  const RecommendationEvidence({
    required this.source,
    required this.dataSourceLabel,
    required this.datasetSampleSize,
    required this.sourcePlatformCounts,
    required this.durationSource,
    required this.durationSampleSize,
    required this.durationSamples,
    required this.exemplarTitles,
    required this.keywordScoreExplanation,
    required this.durationExplanation,
    required this.transcriptSource,
    this.warning,
    this.hookSecondsAnalyzed,
    this.sttFallbackReason,
  });

  final String source;
  final String dataSourceLabel;
  final int datasetSampleSize;
  final Map<String, int> sourcePlatformCounts;
  final String durationSource;
  final int durationSampleSize;
  final List<int> durationSamples;
  final List<String> exemplarTitles;
  final String keywordScoreExplanation;
  final String durationExplanation;
  final String transcriptSource;
  final String? warning;
  final int? hookSecondsAnalyzed;
  final String? sttFallbackReason;

  factory RecommendationEvidence.fromJson(Map<String, dynamic> json) {
    final rawCounts = json['source_platform_counts'];
    return RecommendationEvidence(
      source: json['source']?.toString() ?? '-',
      dataSourceLabel: json['data_source_label']?.toString() ?? '-',
      datasetSampleSize: (json['dataset_sample_size'] as num?)?.toInt() ?? 0,
      sourcePlatformCounts: rawCounts is Map
          ? rawCounts.map(
              (key, value) =>
                  MapEntry(key.toString(), (value as num?)?.toInt() ?? 0),
            )
          : const {},
      durationSource: json['duration_source']?.toString() ?? '-',
      durationSampleSize: (json['duration_sample_size'] as num?)?.toInt() ?? 0,
      durationSamples: (json['duration_samples'] as List<dynamic>? ?? const [])
          .whereType<num>()
          .map((item) => item.toInt())
          .toList(),
      exemplarTitles: (json['exemplar_titles'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      keywordScoreExplanation:
          json['keyword_score_explanation']?.toString() ?? '',
      durationExplanation: json['duration_explanation']?.toString() ?? '',
      transcriptSource: json['transcript_source']?.toString() ?? 'unknown',
      warning: json['warning']?.toString(),
      hookSecondsAnalyzed: (json['hook_seconds_analyzed'] as num?)?.toInt(),
      sttFallbackReason: json['stt_fallback_reason']?.toString(),
    );
  }
}

class AnalysisJobStatus {
  const AnalysisJobStatus({
    required this.status,
    required this.stage,
    required this.progress,
    required this.message,
    this.result,
    this.error,
  });

  final String status;
  final String stage;
  final int progress;
  final String message;
  final AnalysisResultViewData? result;
  final String? error;

  bool get isComplete => status == 'completed';
  bool get isFailed =>
      status == 'failed' || status == 'error' || status == 'not_found';

  factory AnalysisJobStatus.fromJson(Map<String, dynamic> json) {
    final status = json['status']?.toString() ?? 'unknown';
    final rawResult = json['result'];
    return AnalysisJobStatus(
      status: status,
      stage: json['stage']?.toString() ?? status,
      progress: (json['progress'] as num?)?.toInt() ?? 0,
      message: json['message']?.toString() ?? '',
      result: rawResult is Map
          ? AnalysisResultViewData.fromJson(
              Map<String, dynamic>.from(rawResult),
            )
          : null,
      error: json['error']?.toString(),
    );
  }
}
