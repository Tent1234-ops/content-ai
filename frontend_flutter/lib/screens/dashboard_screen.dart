import 'dart:math' as math;
import 'dart:async';

import 'package:flutter/material.dart';

import '../models/dashboard_overview.dart';
import '../repositories/dashboard_repository.dart';
import '../state/auth_scope.dart';
import '../state/theme_controller.dart';
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
  String? _error;
  bool _loading = false;
  List<String> _followedTopics = [];
  bool _showFollowedOnly = false;
  Timer? _pollTimer;

  @override
  void initState() {
    super.initState();
    _load();
    _loadFollowedTopics();
    // Poll dashboard summary every 60s (feature-flagged on backend)
    _pollTimer = Timer.periodic(const Duration(seconds: 60), (_) => _load());
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await _repository.getOverview();
      if (!mounted) return;
      setState(() => _data = data);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _loadFollowedTopics() async {
    try {
      final topics = await _repository.getFollowedTopics();
      if (!mounted) return;
      setState(() => _followedTopics = topics);
    } catch (e) {
      // Silently fail - followed topics are optional
    }
  }

  Future<void> _toggleFollowTopic(String topic) async {
    final isFollowing = _followedTopics.contains(topic);
    try {
      if (isFollowing) {
        await _repository.unfollowTopic(topic);
      } else {
        await _repository.followTopic(topic);
      }
      if (!mounted) return;
      setState(() {
        if (isFollowing) {
          _followedTopics.remove(topic);
        } else {
          _followedTopics.add(topic);
        }
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(isFollowing ? 'Unfollowed $topic' : 'Following $topic')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${e.toString()}')),
        );
      }
    }
  }

  Future<void> _saveIdea(String title, String source) async {
    try {
      await _repository.saveIdea(title, source);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Saved to My Ideas!'), duration: Duration(seconds: 2)),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving idea: ${e.toString()}')),
        );
      }
    }
  }

  void _showTrendDetails(dynamic item, [double? maxScore]) {
    showDialog<void>(
      context: context,
      builder: (context) {
        final score = (item.trendScore ?? 0).toString();
        final platform = item.sourcePlatform ?? '-';
        final label = _confidenceLabel(item.trendScore, maxScore ?? 0);
        return AlertDialog(
          title: Text(item.title ?? 'Trend'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Where: $platform'),
              const SizedBox(height: 8),
              Text('Strength: $label ($score)'),
              const SizedBox(height: 12),
              Text('Summary: This trend is detected from recent performance and engagement signals.'),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Close'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                _saveIdea(item.title ?? 'Trend', platform);
              },
              child: const Text('Save'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _logout() async {
    await AuthScope.of(context).logout();
    if (!mounted) return;
    Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
  }

  String _confidenceLabel(num score, num maxScore) {
    if (maxScore <= 0) return 'Medium';
    final ratio = maxScore > 0 ? score / maxScore : 0;
    if (ratio >= 0.75) return 'High';
    if (ratio >= 0.35) return 'Medium';
    return 'Low';
  }

  Color _confidenceColor(String label, BuildContext context) {
    switch (label) {
      case 'High':
        return Colors.green.shade600;
      case 'Medium':
        return Colors.orange.shade700;
      default:
        return Colors.grey.shade600;
    }
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    final data = _data;
    final metrics = data?.metrics;
    final role = auth.role;
    final comparisons = data?.platformComparison ?? const [];
    final topSources = data?.sourceDistribution ?? const [];
    final topTrends = data?.topTrends ?? const [];

    final liveTrendItems = (data?.liveYoutubeTrends ?? const []).take(20).toList();
    final filteredTrendItems = _showFollowedOnly && _followedTopics.isNotEmpty
        ? topTrends.where((t) => _followedTopics.any((topic) => t.title.toLowerCase().contains(topic.toLowerCase()))).toList()
        : topTrends;
    final maxTrendScore = filteredTrendItems.isNotEmpty
        ? filteredTrendItems.map((t) => t.trendScore.toDouble()).reduce(math.max)
        : 0.0;

    return AppShell(
      title: 'Dashboard',
      currentRoute: '/dashboard',
      isAdmin: auth.isAdmin,
      onLogout: _logout,
      actions: [
        if (auth.isAdmin)
          IconButton(
            onPressed: () => Navigator.pushNamed(context, '/admin-console'),
            icon: const Icon(Icons.admin_panel_settings_outlined),
            tooltip: 'Admin Console',
          ),
        IconButton(
          onPressed: () => Navigator.pushNamed(context, '/upload'),
          icon: const Icon(Icons.upload_file),
          tooltip: 'Analyze My Clip',
        ),
        IconButton(
          onPressed: () => Navigator.pushNamed(context, '/history'),
          icon: const Icon(Icons.history),
          tooltip: 'My History',
        ),
        IconButton(
          onPressed: _load,
          icon: const Icon(Icons.refresh),
          tooltip: 'Refresh',
        ),
      ],
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : data == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _load,
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_loading) const LinearProgressIndicator(),
                      if (role == 'admin')
                        Card(
                          color: Theme.of(context).colorScheme.secondaryContainer,
                          child: ListTile(
                            leading: const Icon(Icons.admin_panel_settings_outlined),
                            title: const Text('Admin tools available'),
                            subtitle: const Text('Open datasets, cluster runs, logs and reporting.'),
                            trailing: const Icon(Icons.chevron_right),
                            onTap: () => Navigator.pushNamed(context, '/admin-console'),
                          ),
                        ),
                      // Key Metrics
                      Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: [
                          _MetricCard(
                            title: 'ข้อมูลชุด',
                            value: '${metrics?.totalDatasetContents ?? 0}',
                            icon: Icons.dataset,
                          ),
                          _MetricCard(
                            title: 'ผู้ใช้',
                            value: '${metrics?.totalUsers ?? 0}',
                            icon: Icons.people,
                          ),
                          _MetricCard(
                            title: 'รอบคลัสเตอร์',
                            value: '${metrics?.totalClusterRuns ?? 0}',
                            icon: Icons.bubble_chart,
                          ),
                          _MetricCard(
                            title: 'การวิเคราะห์ของฉัน',
                            value: '${metrics?.myAnalysisResults ?? 0}',
                            icon: Icons.assessment,
                          ),
                        ],
                      ),
                      const SizedBox(height: 24),

                      // Followed Topics Section
                      _SectionHeader(
                        title: 'หัวข้อที่ติดตาม',
                        subtitle: 'ติดตามเทรนด์ในหมวดที่คุณสนใจ',
                        action: _followedTopics.isEmpty
                            ? null
                            : IconButton(
                                icon: Icon(_showFollowedOnly ? Icons.check_box : Icons.check_box_outline_blank),
                                onPressed: () => setState(() => _showFollowedOnly = !_showFollowedOnly),
                                tooltip: 'กรองตามหัวข้อที่ติดตาม',
                              ),
                      ),
                      if (_followedTopics.isEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              children: [
                                Icon(Icons.bookmark_outline, size: 32, color: Colors.grey.shade400),
                                const SizedBox(height: 8),
                                const Text('ยังไม่ได้ติดตามหัวข้อใด'),
                                const SizedBox(height: 8),
                                const Text(
                                  'แตะไอคอนรูปบุ๊กมาร์กติดกับเทรนด์เพื่อเพิ่มหัวข้อที่สนใจ',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(fontSize: 12, color: Colors.grey),
                                ),
                              ],
                            ),
                          ),
                        )
                      else
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: _followedTopics
                              .map((topic) => Chip(
                                    label: Text(topic),
                                    onDeleted: () => _toggleFollowTopic(topic),
                                    avatar: const Icon(Icons.bookmark, size: 18),
                                  ))
                              .toList(),
                        ),
                      const SizedBox(height: 24),

                      _SectionHeader(
                        title: 'แหล่งข้อมูล',
                        subtitle: 'ข้อมูลที่นำมาวิเคราะห์มาจากแพลตฟอร์มต่าง ๆ',
                      ),
                      if (data.platformSummaries.isEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(
                              'ยังไม่มีข้อมูลจากแหล่งใดในระบบ',
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey),
                            ),
                          ),
                        )
                      else
                        SizedBox(
                          height: 200,
                          child: ListView.separated(
                            scrollDirection: Axis.horizontal,
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            separatorBuilder: (_, __) => const SizedBox(width: 12),
                            itemCount: data.platformSummaries.length,
                            itemBuilder: (context, index) {
                              final item = data.platformSummaries[index];
                              final totalProfiles = item.profileCount;
                              final totalDatasets = item.datasetCount;
                              final domainPreview = item.domains.isEmpty
                                  ? 'ยังไม่มีหมวดข้อมูล'
                                  : item.domains.length <= 3
                                      ? item.domains.join(', ')
                                      : '${item.domains.take(3).join(', ')} +${item.domains.length - 3}';

                              return SizedBox(
                                width: 280,
                                child: Card(
                                  child: Padding(
                                    padding: const EdgeInsets.all(14),
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          children: [
                                            Icon(_getPlatformIcon(item.source), color: Theme.of(context).colorScheme.primary),
                                            const SizedBox(width: 12),
                                            Expanded(
                                              child: Column(
                                                crossAxisAlignment: CrossAxisAlignment.start,
                                                children: [
                                                  Text(
                                                    _formatPlatformName(item.source),
                                                    style: Theme.of(context).textTheme.titleSmall,
                                                  ),
                                                  const SizedBox(height: 4),
                                                  Text(
                                                    '${item.domains.length} หมวดหมู่',
                                                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          ],
                                        ),
                                        const SizedBox(height: 12),
                                        Text(
                                          'ตัวอย่างหมวดข้อมูล',
                                          style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                        ),
                                        const SizedBox(height: 6),
                                        Text(
                                          domainPreview,
                                          maxLines: 2,
                                          overflow: TextOverflow.ellipsis,
                                          style: Theme.of(context).textTheme.bodyMedium,
                                        ),
                                        const Spacer(),
                                        Row(
                                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                          children: [
                                            Column(
                                              crossAxisAlignment: CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  totalDatasets.toString(),
                                                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                                        color: Theme.of(context).primaryColor,
                                                        fontWeight: FontWeight.bold,
                                                      ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'รายการข้อมูล',
                                                  style: Theme.of(context).textTheme.labelSmall,
                                                ),
                                              ],
                                            ),
                                            Column(
                                              crossAxisAlignment: CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  totalProfiles.toString(),
                                                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                                        color: Theme.of(context).colorScheme.secondary,
                                                        fontWeight: FontWeight.bold,
                                                      ),
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'โปรไฟล์ (หมวด)',
                                                  style: Theme.of(context).textTheme.labelSmall,
                                                ),
                                              ],
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      const SizedBox(height: 24),

                      _SectionHeader(
                        title: 'Live YouTube Trends',
                        subtitle: (data?.liveYoutubeTrendMode ?? 'unknown') == 'live'
                            ? 'Realtime trending videos from YouTube'
                            : 'YouTube trends fallback or unavailable',
                      ),
                      if (liveTrendItems.isEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Center(
                              child: Text(
                                'No live YouTube trends available right now.',
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey),
                              ),
                            ),
                          ),
                        )
                      else
                        SizedBox(
                          height: 280,
                          child: ListView.separated(
                            scrollDirection: Axis.horizontal,
                            itemCount: liveTrendItems.length,
                            separatorBuilder: (_, __) => const SizedBox(width: 12),
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            itemBuilder: (context, index) {
                              final item = liveTrendItems[index];
                              final isLive = (data?.liveYoutubeTrendMode ?? 'unknown') == 'live';
                              final sourceLabel = _formatPlatformName(item.sourcePlatform);
                              final strengthLabel = _confidenceLabel(item.trendScore, maxTrendScore);
                              final strengthPercent = maxTrendScore > 0 ? ((item.trendScore.toDouble() / maxTrendScore) * 100).clamp(0.0, 100.0) : 0.0;
                              final strengthPercentLabel = strengthPercent >= 10
                                  ? '${strengthPercent.toStringAsFixed(0)}%'
                                  : '${strengthPercent.toStringAsFixed(1)}%';
                              return _AnimatedEntryCard(
                                index: index,
                                child: SizedBox(
                                  width: 320,
                                  child: Stack(
                                    children: [
                                      Card(
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(24),
                                        ),
                                        elevation: 4,
                                        child: Padding(
                                          padding: const EdgeInsets.all(18),
                                          child: Column(
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                item.title,
                                                maxLines: 4,
                                                overflow: TextOverflow.ellipsis,
                                                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                                      fontWeight: FontWeight.bold,
                                                    ),
                                              ),
                                              const SizedBox(height: 12),
                                              Wrap(
                                                spacing: 8,
                                                runSpacing: 8,
                                                children: [
                                                  _ChipLabel(
                                                    label: isLive ? 'LIVE' : 'FALLBACK',
                                                    color: isLive
                                                        ? Theme.of(context).colorScheme.primary
                                                        : Colors.grey.shade400,
                                                  ),
                                                  _ChipLabel(
                                                    label: sourceLabel,
                                                    color: Theme.of(context).colorScheme.secondary,
                                                  ),
                                                ],
                                              ),
                                              const SizedBox(height: 16),
                                              Text(
                                                'What is this trend?',
                                                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                              ),
                                              const SizedBox(height: 6),
                                              Text(
                                                'เทรนด์นี้มาจากวิดีโอที่มีสัญญาณการมีส่วนร่วมสูงในแพลตฟอร์มนี้',
                                                maxLines: 2,
                                                overflow: TextOverflow.ellipsis,
                                                style: Theme.of(context).textTheme.bodyMedium,
                                              ),
                                              const SizedBox(height: 14),
                                              Row(
                                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                                children: [
                                                  Column(
                                                    crossAxisAlignment: CrossAxisAlignment.start,
                                                    children: [
                                                      Text('Where', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
                                                      const SizedBox(height: 4),
                                                      Text(sourceLabel, style: Theme.of(context).textTheme.bodyMedium),
                                                    ],
                                                  ),
                                                  Column(
                                                    crossAxisAlignment: CrossAxisAlignment.end,
                                                    children: [
                                                      Text('How strong', style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
                                                      const SizedBox(height: 4),
                                                      Text(strengthPercentLabel, style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold)),
                                                    ],
                                                  ),
                                                ],
                                              ),
                                              const SizedBox(height: 14),
                                              LinearProgressIndicator(
                                                value: strengthPercent / 100.0,
                                                minHeight: 8,
                                                color: _confidenceColor(strengthLabel, context),
                                                backgroundColor: Colors.grey.shade300,
                                              ),
                                              const SizedBox(height: 14),
                                              Wrap(
                                                alignment: WrapAlignment.spaceBetween,
                                                spacing: 8,
                                                runSpacing: 8,
                                                children: [
                                                  _MiniActionButton(
                                                    label: _followedTopics.contains(item.title) ? 'เลิกติดตาม' : 'ติดตาม',
                                                    icon: _followedTopics.contains(item.title) ? Icons.bookmark : Icons.bookmark_outline,
                                                    onPressed: () => _toggleFollowTopic(item.title),
                                                  ),
                                                  _MiniActionButton(
                                                    label: 'เก็บ',
                                                    icon: Icons.save_outlined,
                                                    onPressed: () async {
                                                      await _saveIdea(item.title, item.sourcePlatform);
                                                    },
                                                  ),
                                                  _MiniActionButton(
                                                    label: 'รายละเอียด',
                                                    icon: Icons.info_outline,
                                                    onPressed: () => _showTrendDetails(item, maxTrendScore),
                                                  ),
                                                ],
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                      Positioned(
                                        right: 18,
                                        top: 18,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                          decoration: BoxDecoration(
                                            color: Theme.of(context).colorScheme.onPrimary.withOpacity(0.07),
                                            borderRadius: BorderRadius.circular(14),
                                          ),
                                          child: Text(
                                            '${index + 1}',
                                            style: Theme.of(context).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      const SizedBox(height: 24),

                      // Duration Recommendations - ปรับให้มีค่าที่ชัดเจน
                      if (comparisons.isNotEmpty)
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _SectionHeader(
                              title: 'Best Duration by Type',
                              subtitle: 'Recommended length and sample size for top performing categories',
                            ),
                            SizedBox(
                              height: 320,
                              child: ListView.separated(
                                scrollDirection: Axis.horizontal,
                                padding: const EdgeInsets.only(bottom: 8),
                                itemCount: comparisons.take(6).length,
                                separatorBuilder: (_, __) => const SizedBox(width: 12),
                                itemBuilder: (context, index) {
                                  final item = comparisons.take(6).toList()[index];
                                  final youtubeDuration = item.youtubeDuration.isNotEmpty ? item.youtubeDuration : 'N/A';
                                  final googleDuration = item.googleDuration.isNotEmpty ? item.googleDuration : 'N/A';
                                  final totalSamples = item.youtubeSampleSize + item.googleSampleSize;
                                  final maxSamples = totalSamples > 0 ? totalSamples.toDouble() : 1.0;
                                  final ytFactor = item.youtubeSampleSize / maxSamples;
                                  final googleFactor = item.googleSampleSize / maxSamples;
                                  return _AnimatedEntryCard(
                                    index: index,
                                    child: SizedBox(
                                      width: 300,
                                      height: 300,
                                      child: Card(
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(20),
                                        ),
                                        elevation: 3,
                                        child: Padding(
                                          padding: const EdgeInsets.all(18),
                                          child: Column(
                                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Column(
                                                crossAxisAlignment: CrossAxisAlignment.start,
                                                children: [
                                                  Text(
                                                    item.domain,
                                                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                                          fontWeight: FontWeight.bold,
                                                        ),
                                                    maxLines: 2,
                                                    overflow: TextOverflow.ellipsis,
                                                  ),
                                                  const SizedBox(height: 10),
                                                  Row(
                                                    children: [
                                                      _DurationPill(
                                                        label: 'YouTube',
                                                        value: youtubeDuration,
                                                        color: Theme.of(context).colorScheme.primary,
                                                      ),
                                                      const SizedBox(width: 10),
                                                      _DurationPill(
                                                        label: 'Google',
                                                        value: googleDuration,
                                                        color: Theme.of(context).colorScheme.secondary,
                                                      ),
                                                    ],
                                                  ),
                                                  const SizedBox(height: 16),
                                                  Text(
                                                    'Dataset strength',
                                                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                                  ),
                                                  const SizedBox(height: 10),
                                                  Row(
                                                    children: [
                                                      Expanded(
                                                        child: Column(
                                                          crossAxisAlignment: CrossAxisAlignment.start,
                                                          children: [
                                                            Text('YouTube', style: Theme.of(context).textTheme.bodySmall),
                                                            const SizedBox(height: 4),
                                                            LinearProgressIndicator(
                                                              value: ytFactor,
                                                              color: Theme.of(context).colorScheme.primary,
                                                              backgroundColor: Colors.grey.shade200,
                                                              minHeight: 8,
                                                            ),
                                                            const SizedBox(height: 6),
                                                            Text('${item.youtubeSampleSize} vids', style: Theme.of(context).textTheme.labelSmall),
                                                          ],
                                                        ),
                                                      ),
                                                      const SizedBox(width: 12),
                                                      Expanded(
                                                        child: Column(
                                                          crossAxisAlignment: CrossAxisAlignment.start,
                                                          children: [
                                                            Text('Google', style: Theme.of(context).textTheme.bodySmall),
                                                            const SizedBox(height: 4),
                                                            LinearProgressIndicator(
                                                              value: googleFactor,
                                                              color: Theme.of(context).colorScheme.secondary,
                                                              backgroundColor: Colors.grey.shade200,
                                                              minHeight: 8,
                                                            ),
                                                            const SizedBox(height: 6),
                                                            Text('${item.googleSampleSize} vids', style: Theme.of(context).textTheme.labelSmall),
                                                          ],
                                                        ),
                                                      ),
                                                    ],
                                                  ),
                                                ],
                                              ),
                                              Text(
                                                totalSamples > 0
                                                    ? 'Based on $totalSamples sample videos'
                                                    : 'No sample data available yet',
                                                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey.shade600),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ),
                                  );
                                },
                              ),
                            ),
                            const SizedBox(height: 24),
                          ],
                        ),

                      // Dataset-backed trends
                      _SectionHeader(
                        title: 'Trending Now',
                        subtitle: _showFollowedOnly && filteredTrendItems.length < topTrends.length
                            ? 'Saved dataset trends from analysis profiles'
                            : 'Top trends from saved dataset content',
                      ),
                      if (filteredTrendItems.isEmpty)
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Center(
                              child: Text(
                                _showFollowedOnly ? 'No trends in followed topics' : 'No trends available',
                                style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey),
                              ),
                            ),
                          ),
                        )
                      else
                        SizedBox(
                          height: 280,
                          child: ListView.separated(
                            scrollDirection: Axis.horizontal,
                            itemCount: filteredTrendItems.length,
                            separatorBuilder: (_, __) => const SizedBox(width: 12),
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            itemBuilder: (context, index) {
                              final item = filteredTrendItems[index];
                              final isFollowed = _followedTopics.contains(item.title);
                              final trendScore = item.trendScore.toDouble();
                              final trendStrength = maxTrendScore > 0 ? (trendScore / maxTrendScore) * 100.0 : 0.0;
                              final scoreLabel = trendStrength >= 10 ? trendStrength.toStringAsFixed(0) : trendStrength.toStringAsFixed(1);
                              final platformLabel = _formatPlatformName(item.sourcePlatform);
                              final trendLabel = trendStrength >= 75
                                  ? 'Hot trend'
                                  : trendStrength >= 40
                                      ? 'Rising topic'
                                      : 'Watch closely';
 
                              return _AnimatedEntryCard(
                                index: index,
                                child: SizedBox(
                                  width: 320,
                                  child: Stack(
                                    children: [
                                      Card(
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(24),
                                        ),
                                        elevation: 4,
                                        child: Padding(
                                          padding: const EdgeInsets.all(18),
                                          child: Column(
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Row(
                                                children: [
                                                  Expanded(
                                                    child: Column(
                                                      crossAxisAlignment: CrossAxisAlignment.start,
                                                      children: [
                                                        Text(
                                                          item.title,
                                                          maxLines: 3,
                                                          overflow: TextOverflow.ellipsis,
                                                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                                                fontWeight: FontWeight.bold,
                                                              ),
                                                        ),
                                                        const SizedBox(height: 10),
                                                        Wrap(
                                                          spacing: 8,
                                                          runSpacing: 8,
                                                          children: [
                                                            _ChipLabel(
                                                              label: platformLabel,
                                                              color: Theme.of(context).colorScheme.primary,
                                                            ),
                                                            _ChipLabel(
                                                              label: trendLabel,
                                                              color: Theme.of(context).colorScheme.secondary,
                                                            ),
                                                          ],
                                                        ),
                                                      ],
                                                    ),
                                                  ),
                                                  IconButton(
                                                    onPressed: () => _toggleFollowTopic(item.title),
                                                    icon: Icon(
                                                      isFollowed ? Icons.bookmark : Icons.bookmark_outline,
                                                      color: isFollowed ? Theme.of(context).colorScheme.secondary : Colors.grey.shade600,
                                                    ),
                                                    tooltip: isFollowed ? 'Unfollow' : 'Follow',
                                                  ),
                                                ],
                                              ),
                                              const Spacer(),
                                              Text(
                                                'What is this trend?',
                                                style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                              ),
                                              const SizedBox(height: 8),
                                              Text(
                                                'This trend is based on recent performance and engagement signals for similar content.',
                                                maxLines: 2,
                                                overflow: TextOverflow.ellipsis,
                                                style: Theme.of(context).textTheme.bodyMedium,
                                              ),
                                              const SizedBox(height: 16),
                                              Row(
                                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                                crossAxisAlignment: CrossAxisAlignment.center,
                                                children: [
                                                  Column(
                                                    crossAxisAlignment: CrossAxisAlignment.start,
                                                    children: [
                                                      Text(
                                                        '$scoreLabel%',
                                                        style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold),
                                                      ),
                                                      const SizedBox(height: 4),
                                                      Text(
                                                        'Relative strength',
                                                        style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                                                      ),
                                                    ],
                                                  ),
                                                  Container(
                                                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                                                    decoration: BoxDecoration(
                                                      color: _confidenceColor(
                                                                  _confidenceLabel(item.trendScore, maxTrendScore),
                                                                  context)
                                                              .withOpacity(0.14),
                                                      borderRadius: BorderRadius.circular(16),
                                                    ),
                                                    child: Text(
                                                      _confidenceLabel(item.trendScore, maxTrendScore),
                                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                                            color: _confidenceColor(
                                                                _confidenceLabel(item.trendScore, maxTrendScore),
                                                                context),
                                                            fontWeight: FontWeight.w700,
                                                          ),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                      Positioned(
                                        right: 18,
                                        top: 18,
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                          decoration: BoxDecoration(
                                            color: Theme.of(context).colorScheme.onPrimary.withOpacity(0.07),
                                            borderRadius: BorderRadius.circular(14),
                                          ),
                                          child: Text(
                                            '${index + 1}',
                                            style: Theme.of(context).textTheme.bodySmall?.copyWith(fontWeight: FontWeight.bold),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      const SizedBox(height: 24),
                      // Source Distribution Chart
                      if (topSources.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Data Distribution',
                          subtitle: 'Dataset records by source platform',
                        ),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: SimpleBarChart(items: topSources),
                          ),
                        ),
                        const SizedBox(height: 16),
                      ],
                    ],
                  ),
                ),
    );
  }

  IconData _getPlatformIcon(String platform) {
    return switch (platform.toLowerCase()) {
      'youtube' => Icons.play_circle,
      'google' => Icons.search,
      'tiktok' => Icons.videocam,
      'x' || 'twitter' => Icons.message,
      _ => Icons.link,
    };
  }

  String _formatPlatformName(String platform) {
    final normalized = platform.replaceAll('_', ' ').trim();
    return normalized.split(' ').map((part) => part.isEmpty ? part : '${part[0].toUpperCase()}${part.substring(1)}').join(' ');
  }
}

class _ChipLabel extends StatelessWidget {
  const _ChipLabel({
    required this.label,
    required this.color,
  });

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.w600,
          fontSize: 12,
        ),
      ),
    );
  }
}

class _AnimatedEntryCard extends StatelessWidget {
  const _AnimatedEntryCard({
    required this.index,
    required this.child,
  });

  final int index;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: Duration(milliseconds: 400 + (index * 50)),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return Opacity(
          opacity: value,
          child: Transform.translate(
            offset: Offset(0, 20 * (1 - value)),
            child: child,
          ),
        );
      },
      child: child,
    );
  }
}

class _MiniActionButton extends StatelessWidget {
  const _MiniActionButton({
    required this.label,
    required this.icon,
    required this.onPressed,
  });

  final String label;
  final IconData icon;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: FilledButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 16),
        label: Text(label, style: const TextStyle(fontSize: 13)),
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          backgroundColor: Theme.of(context).colorScheme.primary.withOpacity(0.08),
          foregroundColor: Theme.of(context).colorScheme.primary,
          elevation: 0,
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({
    required this.title,
    required this.value,
    required this.icon,
  });

  final String title;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 160,
      child: Card(
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.all(10),
                child: Icon(icon, size: 24, color: Theme.of(context).colorScheme.primary),
              ),
              const SizedBox(height: 12),
              Text(title, style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              Text(value, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
            ],
          ),
        ),
      ),
    );
  }
}

class _DurationPill extends StatelessWidget {
  const _DurationPill({
    required this.label,
    required this.value,
    required this.color,
  });

  final String label;
  final String value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.18)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey)),
          const SizedBox(height: 4),
          Text(value, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold, color: color)),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({
    required this.title,
    required this.subtitle,
    this.action,
  });

  final String title;
  final String subtitle;
  final Widget? action;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                const SizedBox(height: 2),
                Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
          if (action != null) action!,
        ],
      ),
    );
  }
}
