import 'package:flutter/material.dart';

import '../models/admin_report.dart';
import '../models/common_models.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminConsoleScreen extends StatefulWidget {
  const AdminConsoleScreen({super.key});

  @override
  State<AdminConsoleScreen> createState() => _AdminConsoleScreenState();
}

class _AdminConsoleScreenState extends State<AdminConsoleScreen> {
  final _repository = AdminRepository();
  AdminOverviewReport? _overview;
  RecommendationAdminReport? _recommendationReport;
  List<ProfileComparison>? _comparisons;
  String? _error;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final overviewFuture = _repository.getOverviewReport();
      final recommendationFuture = _repository.getRecommendationReport();
      final comparisonFuture = _repository.compareProfiles();
      final results = await Future.wait(
          [overviewFuture, recommendationFuture, comparisonFuture]);
      if (!mounted) return;
      setState(() {
        _overview = results[0] as AdminOverviewReport;
        _recommendationReport = results[1] as RecommendationAdminReport;
        _comparisons = results[2] as List<ProfileComparison>;
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

  @override
  Widget build(BuildContext context) {
    final overview = _overview;
    final recommendationReport = _recommendationReport;
    final comparisons = _comparisons ?? const <ProfileComparison>[];
    final datasetHealth = recommendationReport?.datasetHealth;
    final profileHealth = recommendationReport?.profileHealth;
    final topSources = overview?.topSources ?? const <ChartItem>[];
    final statusBreakdown = overview?.statusBreakdown ?? const <ChartItem>[];
    final topCategories = overview?.topCategories ?? const <ChartItem>[];

    return AppShell(
      title: 'Admin Console',
      currentRoute: '/admin-console',
      isAdmin: true,
      actions: [IconButton(onPressed: _load, icon: const Icon(Icons.refresh))],
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : overview == null || recommendationReport == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _load,
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      if (_loading) const LinearProgressIndicator(),
                      Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: [
                          _AdminMetricCard(
                              title: 'Datasets',
                              value: '${overview.datasetTotal}'),
                          _AdminMetricCard(
                              title: 'Cluster Runs',
                              value: '${overview.clusterRunTotal}'),
                          _AdminMetricCard(
                              title: 'Logs',
                              value: '${overview.systemLogTotal}'),
                          _AdminMetricCard(
                            title: 'Duration Coverage',
                            value:
                                '${datasetHealth?.durationCoverageCount ?? 0}',
                          ),
                          _AdminMetricCard(
                            title: 'YT Profiles',
                            value: '${profileHealth?.youtubeProfiles ?? 0}',
                          ),
                          _AdminMetricCard(
                            title: 'GG Profiles',
                            value: '${profileHealth?.googleProfiles ?? 0}',
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Text('Management',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                      Card(
                        child: Column(
                          children: [
                            ListTile(
                              leading: const Icon(Icons.storage_outlined),
                              title: const Text('Datasets'),
                              subtitle: const Text(
                                  'Browse synced YouTube and Google trend records'),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () => Navigator.pushNamed(
                                  context, '/admin-datasets'),
                            ),
                            ListTile(
                              leading: const Icon(Icons.bubble_chart_outlined),
                              title: const Text('Cluster Runs'),
                              subtitle: const Text(
                                  'Run and inspect KMeans/HDBSCAN cluster jobs'),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () => Navigator.pushNamed(
                                  context, '/admin-clusters'),
                            ),
                            ListTile(
                              leading: const Icon(Icons.receipt_long_outlined),
                              title: const Text('System Logs'),
                              subtitle: const Text(
                                  'Review sync status and backend activity'),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () =>
                                  Navigator.pushNamed(context, '/admin-logs'),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text('Recommendation Core',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              ListTile(
                                contentPadding: EdgeInsets.zero,
                                leading:
                                    const Icon(Icons.account_tree_outlined),
                                title: const Text(
                                    'Classifier -> Keyword Gap Flow'),
                                subtitle: Text(
                                  'Datasets ${datasetHealth?.totalDatasetContents ?? 0} | '
                                  'duration coverage ${(datasetHealth?.durationCoverageRatio ?? 0).toStringAsFixed(2)}',
                                ),
                              ),
                              const SizedBox(height: 8),
                              const Text('YouTube profile domains'),
                              const SizedBox(height: 8),
                              _DomainChips(
                                domains:
                                    profileHealth?.youtubeDomains ?? const [],
                              ),
                              const SizedBox(height: 12),
                              const Text('Google profile domains'),
                              const SizedBox(height: 8),
                              _DomainChips(
                                domains:
                                    profileHealth?.googleDomains ?? const [],
                              ),
                              const SizedBox(height: 16),
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(
                                    child: _ProfilePreview(
                                      title: 'YouTube Profiles',
                                      profiles:
                                          recommendationReport.youtubeProfiles,
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: _ProfilePreview(
                                      title: 'Google Profiles',
                                      profiles:
                                          recommendationReport.googleProfiles,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text('Visual Summary',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Top Sources',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              SimpleBarChart(
                                items: topSources,
                              ),
                              const SizedBox(height: 16),
                              const Text('Top Categories',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              SimpleBarChart(
                                items: topCategories,
                              ),
                              const SizedBox(height: 16),
                              const Text('Log Status Breakdown',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              SimpleBarChart(
                                items: statusBreakdown,
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text('Source Comparison',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                      ...comparisons.take(6).map((item) {
                        return Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  item.domain,
                                  style: const TextStyle(
                                      fontWeight: FontWeight.bold),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  'YouTube ${item.leftSampleSize} | Google ${item.rightSampleSize}',
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  'YT ${item.leftDuration.recommendedRange} | GG ${item.rightDuration.recommendedRange}',
                                ),
                                const SizedBox(height: 12),
                                Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Expanded(
                                      child: _KeywordPreview(
                                        title: 'YouTube keywords',
                                        keywords: item.leftTopKeywords,
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: _KeywordPreview(
                                        title: 'Google keywords',
                                        keywords: item.rightTopKeywords,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        );
                      }),
                    ],
                  ),
                ),
    );
  }
}

class _DomainChips extends StatelessWidget {
  const _DomainChips({required this.domains});

  final List<String> domains;

  @override
  Widget build(BuildContext context) {
    if (domains.isEmpty) {
      return const Text('No domains yet');
    }
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: domains
          .map((domain) => Chip(
                avatar: const Icon(Icons.label_outline, size: 18),
                label: Text(domain),
              ))
          .toList(),
    );
  }
}

class _ProfilePreview extends StatelessWidget {
  const _ProfilePreview({required this.title, required this.profiles});

  final String title;
  final List<DatasetProfile> profiles;

  @override
  Widget build(BuildContext context) {
    final visible = profiles.take(3).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        if (visible.isEmpty)
          const Text('No profiles yet')
        else
          ...visible.map((profile) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(
                '${profile.domain} | samples ${profile.sampleSize} | ${profile.duration.recommendedRange}',
              ),
            );
          }),
      ],
    );
  }
}

class _KeywordPreview extends StatelessWidget {
  const _KeywordPreview({required this.title, required this.keywords});

  final String title;
  final List<KeywordScore> keywords;

  @override
  Widget build(BuildContext context) {
    final labels = keywords.take(4).map((item) => item.keyword).toList();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 6),
        if (labels.isEmpty)
          const Text('-')
        else
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: labels.map((label) => Chip(label: Text(label))).toList(),
          ),
      ],
    );
  }
}

class _AdminMetricCard extends StatelessWidget {
  const _AdminMetricCard({required this.title, required this.value});

  final String title;
  final String value;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 160,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.bodyMedium),
              const SizedBox(height: 8),
              Text(value, style: Theme.of(context).textTheme.headlineSmall),
            ],
          ),
        ),
      ),
    );
  }
}
