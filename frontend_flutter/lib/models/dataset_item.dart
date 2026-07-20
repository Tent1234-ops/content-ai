class DatasetItem {
  const DatasetItem({
    required this.datasetId,
    required this.title,
    required this.sourcePlatform,
    required this.category,
    required this.views,
    required this.likes,
    required this.comments,
    required this.trendScore,
  });

  final int datasetId;
  final String title;
  final String sourcePlatform;
  final String category;
  final int views;
  final int likes;
  final int comments;
  final num trendScore;

  factory DatasetItem.fromJson(Map<String, dynamic> json) {
    return DatasetItem(
      datasetId: (json['dataset_id'] as num?)?.toInt() ?? 0,
      title: json['title']?.toString() ?? '-',
      sourcePlatform: json['source_platform']?.toString() ?? '-',
      category: json['category']?.toString() ?? 'uncategorized',
      views: (json['views'] as num?)?.toInt() ?? 0,
      likes: (json['likes'] as num?)?.toInt() ?? 0,
      comments: (json['comments'] as num?)?.toInt() ?? 0,
      trendScore: json['trend_score'] as num? ?? 0,
    );
  }
}
