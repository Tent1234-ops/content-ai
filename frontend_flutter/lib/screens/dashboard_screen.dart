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
  LiveTrendSnapshot? _snapshot;
  List<FollowedTopicItem> _followedTopics = const [];
  List<NotificationItem> _notifications = const [];
  Timer? _pollTimer;
  String? _error;
  Map<String, String> _sectionErrors = const {};
  String _searchText = '';
  String _platformFilter = 'All';
  String _categoryFilter = 'All';
  String _statusFilter = 'All';
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
    if (_loading) return;
    setState(() {
      _loading = true;
      _error = null;
    });
    final overviewRequest = _capture(_repository.getOverview());
    final snapshotRequest =
        _capture(_repository.getLiveTrendSnapshot(limit: 50));
    final topicsRequest = _capture(_repository.getFollowedTopics());
    final notificationsRequest =
        _capture(_repository.getNotifications(limit: 20));

    final overviewResult = await overviewRequest;
    final snapshotResult = await snapshotRequest;
    final topicsResult = await topicsRequest;
    final notificationsResult = await notificationsRequest;
    if (!mounted) return;

    final errors = <String, String>{};
    if (overviewResult.error != null) {
      errors['Overview'] = overviewResult.error.toString();
    }
    if (snapshotResult.error != null) {
      errors['Live trends'] = snapshotResult.error.toString();
    }
    if (topicsResult.error != null) {
      errors['Followed topics'] = topicsResult.error.toString();
    }
    if (notificationsResult.error != null) {
      errors['Notifications'] = notificationsResult.error.toString();
    }

    final liveSnapshot = snapshotResult.value?.retainPreviousItems(_snapshot);
    setState(() {
      if (liveSnapshot != null) {
        _snapshot = liveSnapshot;
      }
      if (overviewResult.value != null) {
        _data = overviewResult.value;
      } else if (_data == null && liveSnapshot != null) {
        _data = DashboardOverview.liveOnly(liveSnapshot);
      }
      if (topicsResult.value != null) {
        _followedTopics = topicsResult.value!;
      }
      if (notificationsResult.value != null) {
        _notifications = notificationsResult.value!;
      }
      _sectionErrors = errors;
      _error = _data == null
          ? errors['Overview'] ??
              errors['Live trends'] ??
              'Dashboard is unavailable.'
          : null;
      _loading = false;
    });

    if (liveSnapshot != null && liveSnapshot.newCount > 0) {
      await _loadNotificationsOnly();
      if (!mounted) return;
    }

    if (liveSnapshot != null && liveSnapshot.newNotifications.isNotEmpty) {
      final first = liveSnapshot.newNotifications.first;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            liveSnapshot.newCount == 1
                ? first.title
                : '${liveSnapshot.newCount} new live trends detected.',
          ),
        ),
      );
    }
  }

  Future<_DashboardLoadResult<T>> _capture<T>(Future<T> request) async {
    try {
      return _DashboardLoadResult<T>.success(await request);
    } catch (error) {
      return _DashboardLoadResult<T>.failure(error);
    }
  }

  Future<void> _syncTrends() async {
    setState(() => _syncing = true);
    try {
      final results = await _repository.syncAllTrendsLive(limit: 50);
      final fetched =
          results.fold<int>(0, (sum, item) => sum + item.totalFetched);
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

  Future<void> _loadNotificationsOnly() async {
    try {
      final notifications = await _repository.getNotifications(limit: 20);
      if (!mounted) return;
      setState(() {
        _notifications = notifications;
        if (_sectionErrors.containsKey('Notifications')) {
          final errors = Map<String, String>.from(_sectionErrors)
            ..remove('Notifications');
          _sectionErrors = errors;
        }
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _sectionErrors = {
          ..._sectionErrors,
          'Notifications': error.toString(),
        };
      });
    }
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

  List<DashboardPlatformTrends> _displayPlatforms(DashboardOverview data) {
    final snapshot = _snapshot;
    if (snapshot == null) return data.platformTrends;
    final hasSnapshotItems =
        snapshot.platformTrends.any((platform) => platform.items.isNotEmpty);
    return hasSnapshotItems ? snapshot.platformTrends : data.platformTrends;
  }

  List<DashboardTrendItem> _allTrends(DashboardOverview data) {
    return _displayPlatforms(data)
        .expand((platform) => platform.items)
        .toList();
  }

  List<DashboardTrendItem> _filteredTrends(DashboardOverview data) {
    final query = _searchText.trim().toLowerCase();
    return _allTrends(data).where((item) {
      final platformOk = _platformFilter == 'All' ||
          item.sourcePlatform
              .toLowerCase()
              .contains(_platformFilter.toLowerCase());
      final category = item.category.isEmpty ? 'General' : item.category;
      final categoryOk =
          _categoryFilter == 'All' || category == _categoryFilter;
      final statusOk = _statusFilter == 'All' || item.status == _statusFilter;
      final text = '${item.title} ${item.category}'.toLowerCase();
      final searchOk = query.isEmpty || text.contains(query);
      return platformOk && categoryOk && statusOk && searchOk;
    }).toList()
      ..sort((a, b) => b.engagementSignal.compareTo(a.engagementSignal));
  }

  List<String> _categories(DashboardOverview data) {
    final values = _allTrends(data)
        .map((item) => item.category.isEmpty ? 'General' : item.category)
        .where((category) => category != 'All')
        .toSet()
        .toList()
      ..sort();
    return ['All', ...values];
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    final data = _data;
    final unreadNotifications =
        _notifications.where((item) => !item.isRead).length;

    return AppShell(
      title: 'Dashboard',
      currentRoute: '/dashboard',
      isAdmin: auth.isAdmin,
      onLogout: _logout,
      actions: [
        Badge(
          isLabelVisible: unreadNotifications > 0,
          label: Text(
            unreadNotifications > 99 ? '99+' : '$unreadNotifications',
          ),
          child: IconButton(
            onPressed: _loadNotificationsOnly,
            icon: const Icon(Icons.notifications_outlined),
            tooltip: 'Refresh trend notifications',
          ),
        ),
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
                      if (_sectionErrors.isNotEmpty) ...[
                        _DashboardWarning(errors: _sectionErrors),
                        const SizedBox(height: 16),
                      ],
                      _DashboardIntro(
                        totalTrends: _allTrends(data).length,
                        newCount: _snapshot?.newCount ?? 0,
                        generatedAt: _snapshot?.generatedAt ?? '',
                      ),
                      const SizedBox(height: 16),
                      _NotificationPanel(
                        notifications: _notifications,
                        onMarkAllRead: _markAllRead,
                      ),
                      const SizedBox(height: 16),
                      _TrendFilters(
                        platform: _platformFilter,
                        category: _categoryFilter,
                        status: _statusFilter,
                        categories: _categories(data),
                        onSearchChanged: (value) =>
                            setState(() => _searchText = value),
                        onPlatformChanged: (value) =>
                            setState(() => _platformFilter = value),
                        onCategoryChanged: (value) =>
                            setState(() => _categoryFilter = value),
                        onStatusChanged: (value) =>
                            setState(() => _statusFilter = value),
                      ),
                      const SizedBox(height: 16),
                      _TrendDashboardSections(
                        trends: _filteredTrends(data),
                        isFollowing: _isFollowing,
                        onToggleFollow: _toggleFollowTopic,
                      ),
                      const SizedBox(height: 16),
                      for (final platform in _displayPlatforms(data)) ...[
                        _PlatformTrendSection(
                          data: platform,
                          isFollowing: _isFollowing,
                          onToggleFollow: _toggleFollowTopic,
                        ),
                        const SizedBox(height: 16),
                      ],
                      _FollowedTopicsPanel(
                        topics: _followedTopics,
                        onDelete: (topic) async {
                          await _repository.unfollowTopic(topic.id);
                          await _loadFollowStateOnly();
                        },
                      ),
                      const SizedBox(height: 16),
                      if (auth.isAdmin) ...[
                        _MetricsGrid(metrics: data.metrics),
                        const SizedBox(height: 16),
                      ],
                      if (auth.isAdmin && data.sourceDistribution.isNotEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Dataset records by source',
                                  style:
                                      Theme.of(context).textTheme.titleMedium,
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

class _DashboardLoadResult<T> {
  const _DashboardLoadResult._({this.value, this.error});

  factory _DashboardLoadResult.success(T value) {
    return _DashboardLoadResult._(value: value);
  }

  factory _DashboardLoadResult.failure(Object error) {
    return _DashboardLoadResult._(error: error);
  }

  final T? value;
  final Object? error;
}

class _DashboardWarning extends StatelessWidget {
  const _DashboardWarning({required this.errors});

  final Map<String, String> errors;

  @override
  Widget build(BuildContext context) {
    final message = errors.entries
        .map((entry) => '${entry.key}: ${entry.value}')
        .join('\n');
    return Material(
      color: Theme.of(context).colorScheme.errorContainer,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(
              Icons.warning_amber_outlined,
              color: Theme.of(context).colorScheme.onErrorContainer,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                'Some dashboard sections could not refresh. Previous data is still shown.\n$message',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onErrorContainer,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DashboardIntro extends StatelessWidget {
  const _DashboardIntro({
    required this.totalTrends,
    required this.newCount,
    required this.generatedAt,
  });

  final int totalTrends;
  final int newCount;
  final String generatedAt;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            const Icon(Icons.trending_up),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Trending Now',
                      style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 4),
                  Text(
                    '$totalTrends live trends across YouTube, Google, and TikTok',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  if (generatedAt.isNotEmpty)
                    Text(
                      'Last checked ${_formatDateTime(generatedAt)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                ],
              ),
            ),
            _StatusPill(
              label: newCount > 0 ? '$newCount new' : 'Live',
              status: newCount > 0 ? 'Rising' : 'Stable',
            ),
          ],
        ),
      ),
    );
  }
}

class _TrendFilters extends StatelessWidget {
  const _TrendFilters({
    required this.platform,
    required this.category,
    required this.status,
    required this.categories,
    required this.onSearchChanged,
    required this.onPlatformChanged,
    required this.onCategoryChanged,
    required this.onStatusChanged,
  });

  final String platform;
  final String category;
  final String status;
  final List<String> categories;
  final ValueChanged<String> onSearchChanged;
  final ValueChanged<String> onPlatformChanged;
  final ValueChanged<String> onCategoryChanged;
  final ValueChanged<String> onStatusChanged;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Wrap(
          spacing: 12,
          runSpacing: 12,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            SizedBox(
              width: 320,
              child: TextField(
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.search),
                  labelText: 'Search title or category',
                  border: OutlineInputBorder(),
                ),
                onChanged: onSearchChanged,
              ),
            ),
            _FilterDropdown(
              label: 'Platform',
              value: platform,
              values: const ['All', 'youtube', 'google', 'tiktok'],
              onChanged: onPlatformChanged,
            ),
            _FilterDropdown(
              label: 'Category',
              value: category,
              values: categories,
              onChanged: onCategoryChanged,
            ),
            _FilterDropdown(
              label: 'Status',
              value: status,
              values: const ['All', 'Hot', 'Rising', 'Stable', 'Cooling'],
              onChanged: onStatusChanged,
            ),
          ],
        ),
      ),
    );
  }
}

class _FilterDropdown extends StatelessWidget {
  const _FilterDropdown({
    required this.label,
    required this.value,
    required this.values,
    required this.onChanged,
  });

  final String label;
  final String value;
  final List<String> values;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    final safeValue = values.contains(value) ? value : 'All';
    return SizedBox(
      width: 180,
      child: DropdownButtonFormField<String>(
        initialValue: safeValue,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        items: values
            .map((item) => DropdownMenuItem(value: item, child: Text(item)))
            .toList(),
        onChanged: (value) {
          if (value != null) onChanged(value);
        },
      ),
    );
  }
}

class _TrendDashboardSections extends StatelessWidget {
  const _TrendDashboardSections({
    required this.trends,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final List<DashboardTrendItem> trends;
  final bool Function(DashboardTrendItem item) isFollowing;
  final Future<void> Function(DashboardTrendItem item) onToggleFollow;

  @override
  Widget build(BuildContext context) {
    final trendingNow = trends.take(12).toList();
    final rising = trends
        .where((item) =>
            item.status == 'Rising' || item.engagementChangePercent != 0)
        .toList()
      ..sort((a, b) =>
          b.engagementChangePercent.compareTo(a.engagementChangePercent));
    final newItems = trends.where((item) => item.isNew).take(12).toList();
    return Column(
      children: [
        _EngagementChart(trends: trendingNow),
        const SizedBox(height: 16),
        _TrendListSection(
          title: 'Trending Now',
          subtitle: 'Current live trends ranked by engagement signal',
          trends: trendingNow,
          isFollowing: isFollowing,
          onToggleFollow: onToggleFollow,
        ),
        const SizedBox(height: 16),
        _TrendListSection(
          title: 'Fastest Rising',
          subtitle: 'Items moving up compared with the previous snapshot',
          trends: rising.take(8).toList(),
          isFollowing: isFollowing,
          onToggleFollow: onToggleFollow,
        ),
        const SizedBox(height: 16),
        _TrendListSection(
          title: 'New Since You Logged In',
          subtitle: 'Detected after your first live baseline scan',
          trends: newItems,
          emptyText: 'No new live trends detected after your baseline yet.',
          isFollowing: isFollowing,
          onToggleFollow: onToggleFollow,
        ),
        const SizedBox(height: 16),
        _CategoryOverview(trends: trends),
      ],
    );
  }
}

class _EngagementChart extends StatelessWidget {
  const _EngagementChart({required this.trends});

  final List<DashboardTrendItem> trends;

  @override
  Widget build(BuildContext context) {
    final visible = trends.take(8).toList();
    final maxSignal = visible.isEmpty
        ? 0.0
        : visible
            .map((item) => item.engagementSignal.toDouble())
            .reduce(math.max);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Trend engagement',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (visible.isEmpty)
              Text('No live trend data available right now.',
                  style: Theme.of(context).textTheme.bodyMedium)
            else
              ...visible.map((item) {
                final value = maxSignal <= 0
                    ? 0.0
                    : item.engagementSignal.toDouble() / maxSignal;
                return Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 132,
                        child: Text(
                          item.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      Expanded(
                        child: LinearProgressIndicator(
                          value: math.min(1.0, math.max(0.0, value)),
                          minHeight: 10,
                        ),
                      ),
                      const SizedBox(width: 10),
                      _ChangeText(value: item.engagementChangePercent),
                    ],
                  ),
                );
              }),
          ],
        ),
      ),
    );
  }
}

class _TrendListSection extends StatelessWidget {
  const _TrendListSection({
    required this.title,
    required this.subtitle,
    required this.trends,
    required this.isFollowing,
    required this.onToggleFollow,
    this.emptyText = 'No trends match the current filters.',
  });

  final String title;
  final String subtitle;
  final List<DashboardTrendItem> trends;
  final String emptyText;
  final bool Function(DashboardTrendItem item) isFollowing;
  final Future<void> Function(DashboardTrendItem item) onToggleFollow;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 12),
            if (trends.isEmpty)
              Text(emptyText, style: Theme.of(context).textTheme.bodyMedium)
            else
              ...trends.map(
                (item) => _CompactTrendTile(
                  item: item,
                  isFollowing: isFollowing(item),
                  onToggleFollow: () => onToggleFollow(item),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _CompactTrendTile extends StatelessWidget {
  const _CompactTrendTile({
    required this.item,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardTrendItem item;
  final bool isFollowing;
  final VoidCallback onToggleFollow;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Icon(_platformIcon(item.sourcePlatform)),
      title: Text(item.title, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: Text(
        '${_formatPlatformName(item.sourcePlatform)} | ${item.category.isEmpty ? 'General' : item.category}',
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: SizedBox(
        width: 172,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            _ChangeText(value: item.engagementChangePercent),
            const SizedBox(width: 8),
            _StatusPill(label: item.status, status: item.status),
            IconButton(
              onPressed: onToggleFollow,
              icon: Icon(isFollowing ? Icons.bookmark : Icons.bookmark_outline),
              tooltip: isFollowing ? 'Unfollow topic' : 'Follow topic',
            ),
          ],
        ),
      ),
    );
  }
}

class _CategoryOverview extends StatelessWidget {
  const _CategoryOverview({required this.trends});

  final List<DashboardTrendItem> trends;

  @override
  Widget build(BuildContext context) {
    final counts = <String, int>{};
    for (final item in trends) {
      final category = item.category.isEmpty ? 'General' : item.category;
      counts[category] = (counts[category] ?? 0) + 1;
    }
    final items = counts.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('By Category', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (items.isEmpty)
              Text('No categories available.',
                  style: Theme.of(context).textTheme.bodyMedium)
            else
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: items
                    .map((entry) => _SmallPill('${entry.key} (${entry.value})'))
                    .toList(),
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
      _MetricItem(
          'Dataset', metrics.totalDatasetContents, Icons.dataset_outlined),
      _MetricItem('Users', metrics.totalUsers, Icons.people_outline),
      _MetricItem('Cluster runs', metrics.totalClusterRuns,
          Icons.bubble_chart_outlined),
      _MetricItem(
          'My analyses', metrics.myAnalysisResults, Icons.assessment_outlined),
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
                'Live trend alerts will appear here when a new item is detected after your baseline scan.',
                style: Theme.of(context).textTheme.bodyMedium,
              )
            else
              ...visible.map(
                (item) => ListTile(
                  dense: true,
                  contentPadding: EdgeInsets.zero,
                  leading: Icon(
                    item.isRead
                        ? Icons.notifications_none
                        : Icons.notifications_active,
                    color: item.isRead
                        ? Theme.of(context).disabledColor
                        : Theme.of(context).colorScheme.primary,
                  ),
                  title: Text(item.title,
                      maxLines: 1, overflow: TextOverflow.ellipsis),
                  subtitle: Text(
                    '${_formatPlatformName(item.platform)} | ${item.category} | ${_formatDateTime(item.detectedAt)}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  trailing: item.type == 'new_live_trend'
                      ? const Icon(Icons.fiber_new_outlined)
                      : const Icon(Icons.notifications_none),
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
            Text('Followed topics',
                style: Theme.of(context).textTheme.titleMedium),
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
            .map((item) => item.engagementSignal.toDouble())
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
                    'By Platform: ${_formatPlatformName(data.platform)}',
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
    final percent =
        maxScore <= 0 ? 0.0 : item.engagementSignal.toDouble() / maxScore;
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
                  icon: Icon(
                      isFollowing ? Icons.bookmark : Icons.bookmark_outline),
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
                _StatusPill(label: item.status, status: item.status),
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
                _Stat(
                  label: 'Engagement',
                  value: _compactSignal(item.engagementSignal),
                ),
                _Stat(label: 'Views', value: _compactNumber(item.views)),
                Expanded(
                    child: _ChangeText(value: item.engagementChangePercent)),
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
    final color =
        isLive ? Colors.green.shade700 : Theme.of(context).colorScheme.error;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        isLive ? 'LIVE' : mode.toUpperCase(),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.label, required this.status});

  final String label;
  final String status;

  @override
  Widget build(BuildContext context) {
    final color = _statusColor(status);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: Theme.of(context).textTheme.labelSmall?.copyWith(color: color),
      ),
    );
  }
}

class _ChangeText extends StatelessWidget {
  const _ChangeText({required this.value});

  final num value;

  @override
  Widget build(BuildContext context) {
    final change = value.toDouble();
    final color = change > 0
        ? Colors.green.shade700
        : change < 0
            ? Colors.red.shade700
            : Theme.of(context).colorScheme.onSurfaceVariant;
    final prefix = change > 0 ? '+' : '';
    return Text(
      '$prefix${change.toStringAsFixed(0)}%',
      style: Theme.of(context).textTheme.labelMedium?.copyWith(color: color),
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
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
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

Color _statusColor(String status) {
  switch (status.toLowerCase()) {
    case 'hot':
      return Colors.red.shade700;
    case 'rising':
      return Colors.green.shade700;
    case 'cooling':
      return Colors.blue.shade700;
    default:
      return Colors.grey.shade700;
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
      .map((part) =>
          part.isEmpty ? part : '${part[0].toUpperCase()}${part.substring(1)}')
      .join(' ');
}

String _formatDateTime(String value) {
  final parsed = DateTime.tryParse(value);
  if (parsed == null) return value;
  final local = parsed.toLocal();
  final hour = local.hour.toString().padLeft(2, '0');
  final minute = local.minute.toString().padLeft(2, '0');
  return '${local.day}/${local.month} $hour:$minute';
}

String _compactSignal(num value) {
  final numeric = value.toDouble();
  if (numeric >= 1000000) return '${(numeric / 1000000).toStringAsFixed(1)}M';
  if (numeric >= 1000) return '${(numeric / 1000).toStringAsFixed(1)}K';
  return numeric.toStringAsFixed(0);
}

String _compactNumber(int value) {
  if (value >= 1000000) return '${(value / 1000000).toStringAsFixed(1)}M';
  if (value >= 1000) return '${(value / 1000).toStringAsFixed(1)}K';
  return value.toString();
}
