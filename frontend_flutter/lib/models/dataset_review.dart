class DatasetReviewSummary {
  const DatasetReviewSummary({
    required this.total,
    required this.pending,
    required this.approved,
    required this.rejected,
  });

  final int total;
  final int pending;
  final int approved;
  final int rejected;

  factory DatasetReviewSummary.fromJson(Map<String, dynamic> json) {
    return DatasetReviewSummary(
      total: (json['total'] as num?)?.toInt() ?? 0,
      pending: (json['pending'] as num?)?.toInt() ?? 0,
      approved: (json['approved'] as num?)?.toInt() ?? 0,
      rejected: (json['rejected'] as num?)?.toInt() ?? 0,
    );
  }
}

class DatasetReviewRun {
  const DatasetReviewRun({
    required this.collectionRunId,
    required this.datasetVersion,
    required this.status,
    required this.startedAt,
    required this.total,
    required this.pending,
    required this.approved,
    required this.rejected,
  });

  final int collectionRunId;
  final String datasetVersion;
  final String status;
  final DateTime? startedAt;
  final int total;
  final int pending;
  final int approved;
  final int rejected;

  factory DatasetReviewRun.fromJson(Map<String, dynamic> json) {
    return DatasetReviewRun(
      collectionRunId: (json['collection_run_id'] as num?)?.toInt() ?? 0,
      datasetVersion: json['dataset_version']?.toString() ?? '',
      status: json['status']?.toString() ?? '',
      startedAt: DateTime.tryParse(json['started_at']?.toString() ?? ''),
      total: (json['total'] as num?)?.toInt() ?? 0,
      pending: (json['pending'] as num?)?.toInt() ?? 0,
      approved: (json['approved'] as num?)?.toInt() ?? 0,
      rejected: (json['rejected'] as num?)?.toInt() ?? 0,
    );
  }
}

class DatasetReviewTaxonomyLeaf {
  const DatasetReviewTaxonomyLeaf({
    required this.leafKey,
    required this.level1,
    required this.level2,
    required this.level3,
    required this.minimumSampleCount,
    required this.verifiedSampleCount,
    required this.ready,
  });

  final String leafKey;
  final String level1;
  final String level2;
  final String level3;
  final int minimumSampleCount;
  final int verifiedSampleCount;
  final bool ready;

  String get path => '$level1 > $level2 > $level3';

  factory DatasetReviewTaxonomyLeaf.fromJson(Map<String, dynamic> json) {
    return DatasetReviewTaxonomyLeaf(
      leafKey: json['leaf_key']?.toString() ?? '',
      level1: json['category_level_1']?.toString() ?? '',
      level2: json['category_level_2']?.toString() ?? '',
      level3: json['category_level_3']?.toString() ?? '',
      minimumSampleCount: (json['minimum_sample_count'] as num?)?.toInt() ?? 0,
      verifiedSampleCount:
          (json['verified_sample_count'] as num?)?.toInt() ?? 0,
      ready: json['ready'] as bool? ?? false,
    );
  }
}

class DatasetReviewCandidate {
  const DatasetReviewCandidate({
    required this.collectionRunId,
    required this.datasetVersion,
    required this.youtubeId,
    required this.title,
    required this.videoUrl,
    required this.channelTitle,
    required this.proposedLeafKey,
    required this.transcriptLanguage,
    required this.captionType,
    required this.durationSeconds,
    required this.transcript,
    required this.transcriptPreview,
    required this.evidenceTerms,
    required this.automatedChecks,
    required this.views,
    required this.likes,
    required this.comments,
    required this.reviewStatus,
    this.reviewedLeafKey,
    this.transcriptQuality,
    this.reviewer,
    this.reviewedAt,
    this.reviewNotes,
    this.datasetId,
  });

  final int collectionRunId;
  final String datasetVersion;
  final String youtubeId;
  final String title;
  final String videoUrl;
  final String channelTitle;
  final String proposedLeafKey;
  final String transcriptLanguage;
  final String captionType;
  final int durationSeconds;
  final String transcript;
  final String transcriptPreview;
  final List<String> evidenceTerms;
  final Map<String, bool> automatedChecks;
  final int views;
  final int likes;
  final int comments;
  final String reviewStatus;
  final String? reviewedLeafKey;
  final String? transcriptQuality;
  final String? reviewer;
  final DateTime? reviewedAt;
  final String? reviewNotes;
  final int? datasetId;

  bool get allAutomatedChecksPass =>
      automatedChecks.isNotEmpty &&
      automatedChecks.values.every((value) => value);

  factory DatasetReviewCandidate.fromJson(Map<String, dynamic> json) {
    return DatasetReviewCandidate(
      collectionRunId: (json['collection_run_id'] as num?)?.toInt() ?? 0,
      datasetVersion: json['dataset_version']?.toString() ?? '',
      youtubeId: json['source_youtube_id']?.toString() ?? '',
      title: json['title']?.toString() ?? '',
      videoUrl: json['video_url']?.toString() ?? '',
      channelTitle: json['channel_title']?.toString() ?? '',
      proposedLeafKey: json['proposed_leaf_key']?.toString() ?? '',
      transcriptLanguage: json['transcript_language']?.toString() ?? 'und',
      captionType: json['caption_type']?.toString() ?? 'unknown',
      durationSeconds: (json['duration_seconds'] as num?)?.toInt() ?? 0,
      transcript: json['transcript']?.toString() ?? '',
      transcriptPreview: json['transcript_preview']?.toString() ?? '',
      evidenceTerms: (json['evidence_terms'] as List<dynamic>? ?? const [])
          .map((item) => item.toString())
          .toList(),
      automatedChecks: Map<String, bool>.from(
        (json['automated_checks'] as Map?)?.map(
              (key, value) => MapEntry(key.toString(), value == true),
            ) ??
            const {},
      ),
      views: (json['views'] as num?)?.toInt() ?? 0,
      likes: (json['likes'] as num?)?.toInt() ?? 0,
      comments: (json['comments'] as num?)?.toInt() ?? 0,
      reviewStatus: json['review_status']?.toString() ?? 'pending',
      reviewedLeafKey: json['reviewed_leaf_key']?.toString(),
      transcriptQuality: json['transcript_quality']?.toString(),
      reviewer: json['reviewer']?.toString(),
      reviewedAt: DateTime.tryParse(json['reviewed_at']?.toString() ?? ''),
      reviewNotes: json['review_notes']?.toString(),
      datasetId: (json['dataset_id'] as num?)?.toInt(),
    );
  }
}

class DatasetReviewQueueResult {
  const DatasetReviewQueueResult({
    required this.total,
    required this.limit,
    required this.offset,
    required this.summary,
    required this.runs,
    required this.taxonomy,
    required this.items,
  });

  final int total;
  final int limit;
  final int offset;
  final DatasetReviewSummary summary;
  final List<DatasetReviewRun> runs;
  final List<DatasetReviewTaxonomyLeaf> taxonomy;
  final List<DatasetReviewCandidate> items;

  factory DatasetReviewQueueResult.fromJson(Map<String, dynamic> json) {
    return DatasetReviewQueueResult(
      total: (json['total'] as num?)?.toInt() ?? 0,
      limit: (json['limit'] as num?)?.toInt() ?? 0,
      offset: (json['offset'] as num?)?.toInt() ?? 0,
      summary: DatasetReviewSummary.fromJson(
        Map<String, dynamic>.from((json['summary'] as Map?) ?? const {}),
      ),
      runs: (json['runs'] as List<dynamic>? ?? const [])
          .map((item) => DatasetReviewRun.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
      taxonomy: (json['taxonomy'] as List<dynamic>? ?? const [])
          .map((item) => DatasetReviewTaxonomyLeaf.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
      items: (json['items'] as List<dynamic>? ?? const [])
          .map((item) => DatasetReviewCandidate.fromJson(
                Map<String, dynamic>.from(item as Map),
              ))
          .toList(),
    );
  }
}
