class DatasetItem {
  const DatasetItem({
    required this.datasetId,
    required this.title,
    required this.videoUrl,
    required this.transcript,
    required this.sourcePlatform,
    required this.category,
    required this.views,
    required this.likes,
    required this.comments,
    required this.trendScore,
    required this.durationSeconds,
    required this.publishedAt,
  });

  final int datasetId;
  final String title;
  final String videoUrl;
  final String transcript;
  final String sourcePlatform;
  final String category;
  final int views;
  final int likes;
  final int comments;
  final num trendScore;
  final int? durationSeconds;
  final String publishedAt;

  factory DatasetItem.fromJson(Map<String, dynamic> json) {
    return DatasetItem(
      datasetId: (json['dataset_id'] as num?)?.toInt() ?? 0,
      title: json['title']?.toString() ?? '-',
      videoUrl: json['video_url']?.toString() ?? '',
      transcript: json['transcript']?.toString() ?? '',
      sourcePlatform: json['source_platform']?.toString() ?? '-',
      category: json['category']?.toString() ?? 'uncategorized',
      views: (json['views'] as num?)?.toInt() ?? 0,
      likes: (json['likes'] as num?)?.toInt() ?? 0,
      comments: (json['comments'] as num?)?.toInt() ?? 0,
      trendScore: json['trend_score'] as num? ?? 0,
      durationSeconds: (json['duration_seconds'] as num?)?.toInt(),
      publishedAt: json['published_at']?.toString() ?? '',
    );
  }

  Map<String, dynamic> toEditableJson() {
    return {
      'title': title,
      'video_url': videoUrl,
      'transcript': transcript,
      'source_platform': sourcePlatform,
      'category': category,
      'views': views,
      'likes': likes,
      'comments': comments,
      'trend_score': trendScore,
      'duration_seconds': durationSeconds,
    };
  }
}
