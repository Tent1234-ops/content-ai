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

  Future<void> _logout() async {
    await AuthScope.of(context).logout();
    if (!mounted) return;
    Navigator.pushNamedAndRemoveUntil(context, '/login', (_) => false);
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    final data = _data;
    final metrics = data?.metrics;
    final role = auth.role;
    final comparisons = data?.platformComparison ?? const [];
    final topSources = data?.sourceDistribution ?? const [];
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
          ),
        IconButton(
          onPressed: () => Navigator.pushNamed(context, '/upload'),
          icon: const Icon(Icons.upload_file),
        ),
        IconButton(
          onPressed: () => Navigator.pushNamed(context, '/history'),
          icon: const Icon(Icons.history),
        ),
        IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
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
                          color:
                              Theme.of(context).colorScheme.secondaryContainer,
                          child: ListTile(
                            leading:
                                const Icon(Icons.admin_panel_settings_outlined),
                            title: const Text('Admin tools available'),
                            subtitle: const Text(
                                'Open datasets, cluster runs, logs and reporting.'),
                            trailing: const Icon(Icons.chevron_right),
                            onTap: () =>
                                Navigator.pushNamed(context, '/admin-console'),
                          ),
                        ),
                      Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: [
                          _MetricCard(
                              title: 'Datasets',
                              value: '${metrics?.totalDatasetContents ?? 0}'),
                          _MetricCard(
                              title: 'Users',
                              value: '${metrics?.totalUsers ?? 0}'),
                          _MetricCard(
                              title: 'Cluster Runs',
                              value: '${metrics?.totalClusterRuns ?? 0}'),
                          _MetricCard(
                              title: 'My Analyses',
                              value: '${metrics?.myAnalysisResults ?? 0}'),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const _SectionHeader(
                        title: 'Platform Summary',
                        subtitle: 'Snapshot by source and profile coverage',
                      ),
                      ...data.platformSummaries.map((item) {
                        final domains = item.domains.join(', ');
                        return Card(
                          child: ListTile(
                            title: Text(item.source),
                            subtitle: Text(
                              'datasets ${item.datasetCount} | profiles ${item.profileCount}\n'
                              'domains: ${domains.isEmpty ? '-' : domains}',
                            ),
                            isThreeLine: true,
                          ),
                        );
                      }),
                      const SizedBox(height: 16),
                      const _SectionHeader(
                        title: 'Platform Compare',
                        subtitle:
                            'Recommended duration and sample size by domain',
                      ),
                      ...comparisons.take(6).map((item) {
                        return Card(
                          child: ListTile(
                            title: Text(item.domain),
                            subtitle: Text(
                              'YT ${item.youtubeSampleSize} | GG ${item.googleSampleSize}\n'
                              'YT ${item.youtubeDuration} | GG ${item.googleDuration}',
                            ),
                            isThreeLine: true,
                          ),
                        );
                      }),
                      const SizedBox(height: 16),
                      const _SectionHeader(
                        title: 'Top Trends',
                        subtitle:
                            'Highest trend score items in the current dataset',
                      ),
                      ...data.topTrends.map((item) {
                        return Card(
                          child: ListTile(
                            title: Text(item.title),
                            subtitle: Text(
                                '${item.sourcePlatform} | score ${item.trendScore}'),
                          ),
                        );
                      }),
                      const SizedBox(height: 16),
                      const _SectionHeader(
                        title: 'Source Distribution',
                        subtitle:
                            'How many dataset records are stored by source platform',
                      ),
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: SimpleBarChart(
                            items: topSources,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({required this.title, required this.value});

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

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.title, required this.subtitle});

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style:
                  const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
          const SizedBox(height: 2),
          Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
