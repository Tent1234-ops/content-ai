import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/gestures.dart';
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
      final category = _formatTrendCategory(
        item.sourcePlatform,
        item.category,
      );
      final categoryOk =
          _categoryFilter == 'All' || category == _categoryFilter;
      final statusOk = _statusFilter == 'All' || item.status == _statusFilter;
      final text = '${item.title} $category'.toLowerCase();
      final searchOk = query.isEmpty || text.contains(query);
      return platformOk && categoryOk && statusOk && searchOk;
    }).toList()
      ..sort((a, b) => b.engagementSignal.compareTo(a.engagementSignal));
  }

  List<String> _categories(DashboardOverview data) {
    final values = _allTrends(data)
        .map(
          (item) => _formatTrendCategory(
            item.sourcePlatform,
            item.category,
          ),
        )
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
                        generatedAt: _snapshot?.generatedAt ?? '',
                        activePlatforms: _displayPlatforms(data)
                            .where((platform) => platform.items.isNotEmpty)
                            .map((platform) => platform.platform)
                            .toList(),
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
    required this.generatedAt,
    required this.activePlatforms,
  });

  final int totalTrends;
  final String generatedAt;
  final List<String> activePlatforms;

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
                    activePlatforms.isEmpty
                        ? 'No live trends available right now'
                        : '$totalTrends live trends from ${_formatPlatformList(activePlatforms)}',
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
            const _StatusPill(label: 'Live', status: 'Stable'),
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
        child: LayoutBuilder(
          builder: (context, constraints) {
            final stacked = constraints.maxWidth < 760;
            final searchWidth = stacked
                ? constraints.maxWidth
                : math.min(320.0, constraints.maxWidth * 0.34);
            final dropdownWidth = stacked
                ? constraints.maxWidth
                : math.min(
                    240.0,
                    math.max(
                      150.0,
                      (constraints.maxWidth - searchWidth - 36) / 3,
                    ),
                  );
            return Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                SizedBox(
                  width: searchWidth,
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
                  width: dropdownWidth,
                  label: 'Platform',
                  value: platform,
                  values: const ['All', 'youtube', 'google', 'tiktok'],
                  onChanged: onPlatformChanged,
                ),
                _FilterDropdown(
                  width: dropdownWidth,
                  label: 'Category',
                  value: category,
                  values: categories,
                  onChanged: onCategoryChanged,
                ),
                _FilterDropdown(
                  width: dropdownWidth,
                  label: 'Status',
                  value: status,
                  values: const ['All', 'Hot', 'Rising', 'Stable', 'Cooling'],
                  onChanged: onStatusChanged,
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _FilterDropdown extends StatelessWidget {
  const _FilterDropdown({
    required this.width,
    required this.label,
    required this.value,
    required this.values,
    required this.onChanged,
  });

  final double width;
  final String label;
  final String value;
  final List<String> values;
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    final safeValue = values.contains(value) ? value : 'All';
    return SizedBox(
      width: width,
      child: DropdownButtonFormField<String>(
        initialValue: safeValue,
        isExpanded: true,
        menuMaxHeight: 360,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        items: values
            .map(
              (item) => DropdownMenuItem(
                value: item,
                child: Text(
                  item,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            )
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
    final rising = trends.where((item) => item.isMeaningfulRising).toList()
      ..sort((a, b) => b.momentumScore.compareTo(a.momentumScore));
    return Column(
      children: [
        _EngagementChart(trends: trendingNow),
        const SizedBox(height: 16),
        _TrendListSection(
          title: 'Trending Now',
          subtitle: 'Most popular items in the latest live snapshot',
          trends: trendingNow,
          showMovement: false,
          isFollowing: isFollowing,
          onToggleFollow: onToggleFollow,
        ),
        const SizedBox(height: 16),
        _TrendListSection(
          title: 'Fastest Rising',
          subtitle:
              'Items gaining audience activity or provider rank over the comparison window',
          trends: rising.take(8).toList(),
          showMovement: true,
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
    final maxSignalByPlatform = <String, double>{};
    final minRankByPlatform = <String, int>{};
    final maxRankByPlatform = <String, int>{};
    for (final item in visible) {
      final platform = item.sourcePlatform.toLowerCase();
      maxSignalByPlatform[platform] = math.max(
        maxSignalByPlatform[platform] ?? 0,
        item.engagementSignal.toDouble(),
      );
      maxRankByPlatform[platform] = math.max(
        maxRankByPlatform[platform] ?? 0,
        item.rank,
      );
      if (item.rank > 0) {
        minRankByPlatform[platform] = math.min(
          minRankByPlatform[platform] ?? item.rank,
          item.rank,
        );
      }
    }

    double relativeStrength(DashboardTrendItem item) {
      final platform = item.sourcePlatform.toLowerCase();
      final hasVideoMetrics =
          item.views > 0 || item.likes > 0 || item.comments > 0;
      final maxSignal = maxSignalByPlatform[platform] ?? 0;
      if (hasVideoMetrics && maxSignal > 0) {
        final ratio = item.engagementSignal.toDouble() / maxSignal;
        return math.sqrt(math.max(0.0, ratio)).clamp(0.0, 1.0).toDouble();
      }

      final rank = item.rank > 0 ? item.rank : 1;
      final minRank = minRankByPlatform[platform] ?? 1;
      final maxRank = math.max(maxRankByPlatform[platform] ?? 1, 1);
      if (maxRank <= minRank) return 1;
      final rankPosition = (rank - minRank) / (maxRank - minRank);
      return (1 - (rankPosition * 0.75)).clamp(0.25, 1.0).toDouble();
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Current trend strength',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              'Bar length is relative within each platform. Movement appears only when a previous live value can be compared.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            if (visible.isEmpty)
              Text('No live trend data available right now.',
                  style: Theme.of(context).textTheme.bodyMedium)
            else
              ...visible.map((item) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        item.title,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 6),
                      Wrap(
                        spacing: 8,
                        runSpacing: 6,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: [
                          _SmallPill(_formatPlatformName(item.sourcePlatform)),
                          _SmallPill(
                            _formatTrendCategory(
                              item.sourcePlatform,
                              item.category,
                            ),
                          ),
                          _StatusPill(label: item.status, status: item.status),
                          if (_hasUsefulMovement(item)) _ChangeText(item: item),
                        ],
                      ),
                      const SizedBox(height: 8),
                      LinearProgressIndicator(
                        value: relativeStrength(item),
                        minHeight: 10,
                      ),
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
    required this.showMovement,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final String title;
  final String subtitle;
  final List<DashboardTrendItem> trends;
  final bool showMovement;
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
              Text(
                'No trends match the current filters.',
                style: Theme.of(context).textTheme.bodyMedium,
              )
            else
              ...trends.map(
                (item) => _CompactTrendTile(
                  item: item,
                  showMovement: showMovement,
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
    required this.showMovement,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardTrendItem item;
  final bool showMovement;
  final bool isFollowing;
  final VoidCallback onToggleFollow;

  @override
  Widget build(BuildContext context) {
    if (MediaQuery.sizeOf(context).width < 700) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Icon(_platformIcon(item.sourcePlatform)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(item.title,
                      maxLines: 2, overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 2),
                  Text(
                    '${_formatPlatformName(item.sourcePlatform)} | ${_formatTrendCategory(item.sourcePlatform, item.category)}',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      if (showMovement && _hasUsefulMovement(item))
                        _ChangeText(item: item),
                      _StatusPill(label: item.status, status: item.status),
                    ],
                  ),
                ],
              ),
            ),
            IconButton(
              onPressed: onToggleFollow,
              icon: Icon(isFollowing ? Icons.bookmark : Icons.bookmark_outline),
              tooltip: isFollowing ? 'Unfollow topic' : 'Follow topic',
            ),
          ],
        ),
      );
    }
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Icon(_platformIcon(item.sourcePlatform)),
      title: Text(item.title, maxLines: 1, overflow: TextOverflow.ellipsis),
      subtitle: Text(
        '${_formatPlatformName(item.sourcePlatform)} | ${_formatTrendCategory(item.sourcePlatform, item.category)}',
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: SizedBox(
        width: 220,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            if (showMovement && _hasUsefulMovement(item)) ...[
              _ChangeText(item: item),
              const SizedBox(width: 8),
            ],
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
      final category = _formatTrendCategory(
        item.sourcePlatform,
        item.category,
      );
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
                    '${_formatPlatformName(item.platform)} | ${_formatTrendCategory(item.platform, item.category)} | ${_formatDateTime(item.detectedAt)}',
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

class _PlatformTrendSection extends StatefulWidget {
  const _PlatformTrendSection({
    required this.data,
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardPlatformTrends data;
  final bool Function(DashboardTrendItem item) isFollowing;
  final Future<void> Function(DashboardTrendItem item) onToggleFollow;

  @override
  State<_PlatformTrendSection> createState() => _PlatformTrendSectionState();
}

class _PlatformTrendSectionState extends State<_PlatformTrendSection> {
  final ScrollController _scrollController = ScrollController();

  Future<void> _scrollBy(double direction) async {
    if (!_scrollController.hasClients) return;
    final position = _scrollController.position;
    final distance = math.max(280.0, position.viewportDimension * 0.85);
    final target = (position.pixels + (distance * direction))
        .clamp(position.minScrollExtent, position.maxScrollExtent)
        .toDouble();
    await _scrollController.animateTo(
      target,
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutCubic,
    );
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final data = widget.data;
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
                IconButton(
                  onPressed: data.items.isEmpty ? null : () => _scrollBy(-1),
                  icon: const Icon(Icons.chevron_left),
                  tooltip: 'Previous trends',
                ),
                IconButton(
                  onPressed: data.items.isEmpty ? null : () => _scrollBy(1),
                  icon: const Icon(Icons.chevron_right),
                  tooltip: 'More trends',
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
                height: 210,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final visibleCards = math.max(
                      1,
                      math.min(
                        6,
                        ((constraints.maxWidth + 12) / 292).floor(),
                      ),
                    );
                    final cardWidth =
                        (constraints.maxWidth - 8 - (12 * (visibleCards - 1))) /
                            visibleCards;
                    return Scrollbar(
                      controller: _scrollController,
                      thumbVisibility: true,
                      trackVisibility: true,
                      interactive: true,
                      scrollbarOrientation: ScrollbarOrientation.bottom,
                      child: ScrollConfiguration(
                        behavior: ScrollConfiguration.of(context).copyWith(
                          dragDevices: const {
                            PointerDeviceKind.touch,
                            PointerDeviceKind.mouse,
                            PointerDeviceKind.trackpad,
                          },
                        ),
                        child: ListView.separated(
                          controller: _scrollController,
                          scrollDirection: Axis.horizontal,
                          padding: const EdgeInsets.only(bottom: 14, right: 8),
                          itemCount: data.items.length,
                          separatorBuilder: (_, __) =>
                              const SizedBox(width: 12),
                          itemBuilder: (context, index) {
                            final item = data.items[index];
                            return SizedBox(
                              width: cardWidth,
                              child: _TrendCard(
                                item: item,
                                rank: item.rank > 0 ? item.rank : index + 1,
                                isFollowing: widget.isFollowing(item),
                                onToggleFollow: () =>
                                    widget.onToggleFollow(item),
                              ),
                            );
                          },
                        ),
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
    required this.isFollowing,
    required this.onToggleFollow,
  });

  final DashboardTrendItem item;
  final int rank;
  final bool isFollowing;
  final VoidCallback onToggleFollow;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
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
                    maxLines: 4,
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
                _SmallPill(
                  _formatTrendCategory(
                    item.sourcePlatform,
                    item.category,
                  ),
                ),
                _StatusPill(label: item.status, status: item.status),
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
  const _ChangeText({required this.item});

  final DashboardTrendItem item;

  @override
  Widget build(BuildContext context) {
    final color = _movementColor(context, item);
    return Tooltip(
      message: _momentumTooltip(item),
      child: Text(
        item.changeLabel,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
        textAlign: TextAlign.right,
        style: Theme.of(context).textTheme.labelMedium?.copyWith(color: color),
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
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(label, style: Theme.of(context).textTheme.labelSmall),
    );
  }
}

bool _hasUsefulMovement(DashboardTrendItem item) {
  if (item.changeKind == 'new') return true;
  if (item.changeKind == 'velocity_up') return item.isMeaningfulRising;
  if (item.changeKind == 'velocity_down') {
    return item.status.toLowerCase() == 'cooling';
  }
  if (item.changeKind == 'rank_up' || item.changeKind == 'rank_down') {
    return item.rankChange.abs() >= 2;
  }
  return item.changeKind == 'interest' &&
      item.engagementChangePercent.abs() >= 0.1;
}

Color _movementColor(BuildContext context, DashboardTrendItem item) {
  final kind = item.changeKind;
  final rising = kind == 'velocity_up' ||
      kind == 'rank_up' ||
      (kind == 'interest' && item.engagementChangePercent > 0) ||
      kind == 'new';
  final cooling = kind == 'velocity_down' ||
      kind == 'rank_down' ||
      (kind == 'interest' && item.engagementChangePercent < 0);
  if (rising) return Colors.green.shade700;
  if (cooling) return Colors.red.shade700;
  return Theme.of(context).colorScheme.onSurfaceVariant;
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
  return normalized.split(' ').map((part) {
    switch (part.toLowerCase()) {
      case 'youtube':
        return 'YouTube';
      case 'tiktok':
        return 'TikTok';
      default:
        return part.isEmpty
            ? part
            : '${part[0].toUpperCase()}${part.substring(1)}';
    }
  }).join(' ');
}

String _formatPlatformList(List<String> platforms) {
  final names = platforms.map(_formatPlatformName).toList();
  if (names.isEmpty) return '';
  if (names.length == 1) return names.first;
  if (names.length == 2) return '${names.first} and ${names.last}';
  return '${names.sublist(0, names.length - 1).join(', ')}, and ${names.last}';
}

String _formatTrendCategory(String platform, String category) {
  final value = category.trim();
  final normalized = value.toLowerCase();
  final isGoogle = platform.toLowerCase().contains('google');
  final looksLikeRegion = value.length == 2 &&
      value.codeUnits.every(
        (character) =>
            (character >= 65 && character <= 90) ||
            (character >= 97 && character <= 122),
      );
  if (isGoogle &&
      (value.isEmpty ||
          looksLikeRegion ||
          {'general', 'search', 'thailand'}.contains(normalized))) {
    return 'Search Trends';
  }
  if (value.isEmpty || normalized == 'general') return 'General';
  return value;
}

String _formatDateTime(String value) {
  final trimmed = value.trim();
  if (trimmed.isEmpty) return value;
  final hasTimezone = RegExp(
    r'(z|[+-]\d{2}:?\d{2})$',
    caseSensitive: false,
  ).hasMatch(trimmed);
  final parsed = DateTime.tryParse(hasTimezone ? trimmed : '${trimmed}Z');
  if (parsed == null) return value;
  final local = parsed.toLocal();
  final hour = local.hour.toString().padLeft(2, '0');
  final minute = local.minute.toString().padLeft(2, '0');
  return '${local.day}/${local.month} $hour:$minute';
}

String _momentumTooltip(DashboardTrendItem item) {
  final seconds = item.comparisonWindowSeconds;
  final window = seconds > 0 ? ' Comparison window: ${seconds}s.' : '';
  switch (item.changeKind) {
    case 'metric_baseline':
      return 'YouTube changed the public view-count definition. This snapshot starts a new baseline and is not compared with the previous metric version.';
    case 'velocity_up':
    case 'velocity_down':
      return 'Audience activity compared over a stable window using views, likes, and comments. '
          'This is not a views-per-minute count.$window';
    case 'rank_up':
    case 'rank_down':
      return 'Position change in the live provider ranking since the previous snapshot.$window';
    case 'interest':
      return 'Change in the provider interest signal since the previous snapshot.$window';
    case 'new':
      return 'First appearance in the current live snapshot.';
    case 'none':
      return 'No measurable movement since the previous snapshot.$window';
    default:
      return 'Waiting for a previous snapshot to calculate movement.';
  }
}
