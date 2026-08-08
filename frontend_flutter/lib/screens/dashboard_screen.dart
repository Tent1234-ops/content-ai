import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/dashboard_overview.dart';
import '../repositories/dashboard_repository.dart';
import '../state/auth_scope.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final _repository = DashboardRepository();
  DashboardOverview? _data;
  List<FollowedTopicItem> _followedTopics = const [];
  List<NotificationItem> _notifications = const [];
  Timer? _pollTimer;
  String? _error;
  bool _loading = false;
  bool _syncing = false;

  @override
  void initState() {
    super.initState();
    _loadAll();
    _pollTimer = Timer.periodic(const Duration(seconds: 60), (_) => _loadAll());
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final overview = await _repository.getOverview();
      final followedTopics = await _repository.getFollowedTopics();
      final notifications = await _repository.getNotifications(limit: 20);
      if (!mounted) return;
      setState(() {
        _data = overview;
        _followedTopics = followedTopics;
        _notifications = notifications;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _syncTrends() async {
    setState(() => _syncing = true);
    try {
      final results = await _repository.syncAllTrendsLive(limit: 20);
      final fetched = results.fold<int>(0, (sum, item) => sum + item.totalFetched);
      final saved = results.fold<int>(
        0,
        (sum, item) => sum + item.created + item.updated,
      );
      final notifications = results.fold<int>(
        0,
        (sum, item) => sum + item.notifications,
      );
      final failed = results.where((item) => item.failed).length;
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            failed == 0
                ? 'Live sync complete: fetched $fetched, saved $saved, notifications $notifications.'
                : 'Live sync finished with $failed failed source(s): fetched $fetched, saved $saved.',
          ),
        ),
      );
      await _loadAll();
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Live sync failed: $error')),
      );
    } finally {
      if (mounted) {
        setState(() => _syncing = false);
      }
    }
  }

  Future<void> _toggleFollowTopic(DashboardTrendItem item) async {
    final value = item.title.trim().toLowerCase();
    final existing = _followedTopics.where(
      (topic) => topic.matchType == 'keyword' && topic.value == value,
    );
    try {
      if (existing.isNotEmpty) {
        await _repository.unfollowTopic(existing.first.id);
      } else {
        await _repository.followTopic(item.title);
      }
      await _loadFollowStateOnly();
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Follow topic failed: $error')),
      );
    }
  }

  Future<void> _loadFollowStateOnly() async {
    final topics = await _repository.getFollowedTopics();
    if (!mounted) return;
    setState(() => _followedTopics = topics);
  }

  Future<void> _markAllRead() async {
    final unreadIds = _notifications
        .where((item) => !item.isRead)
        .map((item) => item.id)
        .where((id) => id > 0)
        .toList();
    if (unreadIds.isEmpty) return;
    await _repository.markNotificationsRead(unreadIds);
    final notifications = await _repository.getNotifications(limit: 20);
    if (!mounted) return;
    setState(() => _notifications = notifications);
  }

  Future<void> _logout() async {
    await AuthScope.of(context).logout();
    if (!mounted) return;
    Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
  }

  bool _isFollowing(DashboardTrendItem item) {
    final value = item.title.trim().toLowerCase();
    return _followedTopics.any(
      (topic) => topic.matchType == 'keyword' && topic.value == value,
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    final data = _data;

    return AppShell(
      title: 'Dashboard',
      currentRoute: '/dashboard',
      isAdmin: auth.isAdmin,
      onLogout: _logout,
      actions: [
        if (auth.isAdmin)
          IconButton(
            onPressed: _syncing ? null : _syncTrends,
            icon: _syncing
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.cloud_sync_outlined),
            tooltip: 'Sync live trends',
          ),
        IconButton(
          onPressed: _loadAll,
          icon: const Icon(Icons.refresh),
          tooltip: 'Refresh',
        ),
        IconButton(
          onPressed: () => Navigator.pushNamed(context, '/upload'),
          icon: const Icon(Icons.upload_file),
          tooltip: 'Analyze My Clip',
        ),
      ],
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _loadAll)
          : data == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _loadAll,
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_loading) const LinearProgressIndicator(),
                      _MetricsGrid(metrics: data.metrics),
                      const SizedBox(height: 16),
                      _NotificationPanel(
                        notifications: _notifications,
                        onMarkAllRead: _markAllRead,
                      ),
                      const SizedBox(height: 16),
                      _FollowedTopicsPanel(
                        topics: _followedTopics,
                        onDelete: (topic) async {
                          await _repository.unfollowTopic(topic.id);
                          await _loadFollowStateOnly();
                        },
                      ),
                      const SizedBox(height: 16),
                      for (final platform in data.platformTrends) ...[
                        _PlatformTrendSection(
                          data: platform,
                          isFollowing: _isFollowing,
                          onToggleFollow: _toggleFollowTopic,
                        ),
                        const SizedBox(height: 16),
                      ],
                      if (data.sourceDistribution.isNotEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Dataset records by source',
                                  style: Theme.of(context).textTheme.titleMedium,
                                ),
                                const SizedBox(height: 12),
                                SimpleBarChart(items: data.sourceDistribution),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
    );
  }
}

class _MetricsGrid extends StatelessWidget {
  const _MetricsGrid({required this.metrics});

  final DashboardMetrics metrics;

  @override
  Widget build(BuildContext context) {
    final items = [
      _MetricItem('Dataset', metrics.totalDatasetContents, Icons.dataset_outlined),
      _MetricItem('Users', metrics.totalUsers, Icons.people_outline),
      _MetricItem('Cluster runs', metrics.totalClusterRuns, Icons.bubble_chart_outlined),
      _MetricItem('My analyses', metrics.myAnalysisResults, Icons.assessment_outlined),
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final wide = constraints.maxWidth >= 720;
        return GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: items.length,
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: wide ? 4 : 2,
            mainAxisExtent: 104,
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
          ),
          itemBuilder: (context, index) => _MetricCard(item: items[index]),
        );
      },
    );
  }
}

class _MetricItem {
  const _MetricItem(this.label, this.value, this.icon);

  final String label;
  final int value;
  final IconData icon;
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({required this.item});

  final _MetricItem item;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            Icon(item.icon, color: Theme.of(context).colorScheme.primary),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.value.toString(),
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  Text(
                    item.label,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _NotificationPanel extends StatelessWidget {
  const _NotificationPanel({
    required this.notifications,
    required this.onMarkAllRead,
  });

  final List<NotificationItem> notifications;
  final VoidCallback onMarkAllRead;

  @override
  Widget build(BuildContext context) {
    final unread = notifications.where((item) => !item.isRead).length;
    final visible = notifications.take(4).toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.notifications_outlined),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Trend notifications',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                if (unread > 0)
                  TextButton.icon(
                    onPressed: onMarkAllRead,
                    icon: const Icon(Icons.done_all),
                    label: Text('Mark $unread read'),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            if (visible.isEmpty)
              Text(
                'Follow a topic, then sync live trends to receive matching alerts.',
                style: Theme.of(context).textTheme.bodyMedium,
              )
            else
              ...visible.map(
                (item) => ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    item.isRead ? Icons.notifications_none : Icons.notifications_active,
                    color: item.isRead
                        ? Theme.of(context).disabledColor
                        : Theme.of(context).colorScheme.primary,
                  ),
                  title: Text(item.title, maxLines: 1, overflow: TextOverflow.ellipsis),
                  subtitle: Text(
                    '${_formatPlatformName(item.sourcePlatform)} | ${item.topic}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  trailing: Text(item.trendScore.toStringAsFixed(0)),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _FollowedTopicsPanel extends StatelessWidget {
  const _FollowedTopicsPanel({
    required this.topics,
    required this.onDelete,
  });

  final List<FollowedTopicItem> topics;
  final Future<void> Function(FollowedTopicItem topic) onDelete;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Followed topics', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (topics.isEmpty)
              Text(
                'Tap the bookmark button on any live trend to follow it.',
                style: Theme.of(context).textTheme.bodyMedium,
              )
            else
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: topics
                    .map(
                      (topic) => InputChip(
                        avatar: const Icon(Icons.bookmark, size: 18),
                        label: Text(topic.value),
                        onDeleted: () => onDelete(topic),
                      ),
                    )
                    .toList(),
              ),
          ],
        ),
      ),
    );
  }
}

class _PlatformTrendSection extends StatelessWidget {
  const _PlatformTrendSection({
    required this.data,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardPlatformTrends data;
  final bool Function(DashboardTrendItem item) isFollowing;
  final Future<void> Function(DashboardTrendItem item) onToggleFollow;

  @override
  Widget build(BuildContext context) {
    final maxScore = data.items.isEmpty
        ? 0.0
        : data.items
            .map((item) => item.trendScore.toDouble())
            .reduce(math.max);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_platformIcon(data.platform)),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '${_formatPlatformName(data.platform)} live trends',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                _ModePill(mode: data.mode),
              ],
            ),
            const SizedBox(height: 12),
            if (data.items.isEmpty)
              Text(
                data.mode == 'live_error'
                    ? 'Live source is unavailable right now. No mock fallback is shown.'
                    : 'No live trends returned right now.',
                style: Theme.of(context).textTheme.bodyMedium,
              )
            else
              SizedBox(
                height: 260,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: data.items.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 12),
                  itemBuilder: (context, index) {
                    final item = data.items[index];
                    return SizedBox(
                      width: 300,
                      child: _TrendCard(
                        item: item,
                        rank: index + 1,
                        maxScore: maxScore,
                        isFollowing: isFollowing(item),
                        onToggleFollow: () => onToggleFollow(item),
                      ),
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _TrendCard extends StatelessWidget {
  const _TrendCard({
    required this.item,
    required this.rank,
    required this.maxScore,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardTrendItem item;
  final int rank;
  final double maxScore;
  final bool isFollowing;
  final VoidCallback onToggleFollow;

  @override
  Widget build(BuildContext context) {
    final percent = maxScore <= 0 ? 0.0 : item.trendScore.toDouble() / maxScore;
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '#$rank',
                  style: Theme.of(context).textTheme.labelLarge,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    item.title,
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                ),
                IconButton(
                  onPressed: onToggleFollow,
                  icon: Icon(isFollowing ? Icons.bookmark : Icons.bookmark_outline),
                  tooltip: isFollowing ? 'Unfollow topic' : 'Follow topic',
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                if (item.category.isNotEmpty) _SmallPill(item.category),
                _SmallPill(_formatPlatformName(item.sourcePlatform)),
              ],
            ),
            const Spacer(),
            LinearProgressIndicator(
              value: math.min(1.0, math.max(0.0, percent)),
              minHeight: 8,
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                _Stat(label: 'Score', value: item.trendScore.toStringAsFixed(0)),
                _Stat(label: 'Views', value: _compactNumber(item.views)),
                _Stat(label: 'Likes', value: _compactNumber(item.likes)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ModePill extends StatelessWidget {
  const _ModePill({required this.mode});

  final String mode;

  @override
  Widget build(BuildContext context) {
    final isLive = mode == 'live';
    final color = isLive ? Colors.green.shade700 : Theme.of(context).colorScheme.error;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        isLive ? 'LIVE' : mode.toUpperCase(),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
      ),
    );
  }
}

class _SmallPill extends StatelessWidget {
  const _SmallPill(this.label);

  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceVariant,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(label, style: Theme.of(context).textTheme.labelSmall),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(value, style: Theme.of(context).textTheme.labelLarge),
          Text(label, style: Theme.of(context).textTheme.labelSmall),
        ],
      ),
    );
  }
}

IconData _platformIcon(String platform) {
  final normalized = platform.toLowerCase();
  if (normalized.contains('youtube')) return Icons.play_circle_outline;
  if (normalized.contains('google')) return Icons.search;
  if (normalized.contains('tiktok')) return Icons.music_video_outlined;
  return Icons.public;
}

String _formatPlatformName(String platform) {
  final normalized = platform.replaceAll('_', ' ').trim();
  if (normalized.isEmpty) return '-';
  return normalized
      .split(' ')
      .map((part) => part.isEmpty
          ? part
          : '${part[0].toUpperCase()}${part.substring(1)}')
      .join(' ');
}

String _compactNumber(int value) {
  if (value >= 1000000) return '${(value / 1000000).toStringAsFixed(1)}M';
  if (value >= 1000) return '${(value / 1000).toStringAsFixed(1)}K';
  return value.toString();
}
