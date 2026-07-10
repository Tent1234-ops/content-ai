class SystemLogItem {
  const SystemLogItem({
    required this.logId,
    required this.action,
    required this.status,
    required this.detail,
  });

  final int logId;
  final String action;
  final String status;
  final String detail;

  factory SystemLogItem.fromJson(Map<String, dynamic> json) {
    return SystemLogItem(
      logId: (json['log_id'] as num?)?.toInt() ?? 0,
      action: json['action']?.toString() ?? '-',
      status: json['status']?.toString() ?? 'unknown',
      detail: json['detail']?.toString() ?? '-',
    );
  }
}
