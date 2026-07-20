class KeywordScore {
  const KeywordScore({required this.keyword, required this.score});

  final String keyword;
  final double score;

  factory KeywordScore.fromJson(Map<String, dynamic> json) {
    return KeywordScore(
      keyword: json['keyword']?.toString() ?? '-',
      score: (json['score'] as num?)?.toDouble() ?? 0,
    );
  }
}

class ChartItem {
  const ChartItem({required this.label, required this.count});

  final String label;
  final num count;

  factory ChartItem.fromJson(
    Map<String, dynamic> json, {
    required String labelKey,
    String valueKey = 'count',
  }) {
    return ChartItem(
      label: json[labelKey]?.toString() ?? '-',
      count: json[valueKey] as num? ?? 0,
    );
  }

  Map<String, dynamic> toChartJson(String labelKey, String valueKey) {
    return {labelKey: label, valueKey: count};
  }
}

class PaginatedResult<T> {
  const PaginatedResult({
    required this.total,
    required this.items,
  });

  final int total;
  final List<T> items;
}
