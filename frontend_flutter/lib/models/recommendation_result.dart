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
    required this.missingKeywords,
    required this.hookKeywords,
    required this.missingDimensions,
    required this.duration,
    this.classification,
  });

  final String domain;
  final List<KeywordScore> missingKeywords;
  final List<KeywordScore> hookKeywords;
  final List<MissingDimension> missingDimensions;
  final DurationRecommendation duration;
  final ClassificationResult? classification;

  factory RecommendationResult.fromJson(Map<String, dynamic> json) {
    return RecommendationResult(
      domain: json['domain']?.toString() ?? '-',
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
    required this.title,
    required this.transcript,
    required this.recommendation,
    required this.fallbackDomain,
    required this.raw,
  });

  final String title;
  final String transcript;
  final RecommendationResult recommendation;
  final String fallbackDomain;
  final Map<String, dynamic> raw;

  factory AnalysisResultViewData.fromJson(Map<String, dynamic> json) {
    final analysisRoot =
        Map<String, dynamic>.from((json['analysis'] as Map?) ?? const {});
    final analysis = Map<String, dynamic>.from(
        (analysisRoot['analysis'] as Map?) ?? const {});
    return AnalysisResultViewData(
      title: json['title']?.toString() ??
          analysis['title']?.toString() ??
          'Result',
      transcript: json['transcript']?.toString() ?? '',
      recommendation: RecommendationResult.fromJson(
        Map<String, dynamic>.from((json['recommendation'] as Map?) ?? const {}),
      ),
      fallbackDomain: analysis['domain']?.toString() ?? '-',
      raw: json,
    );
  }
}
