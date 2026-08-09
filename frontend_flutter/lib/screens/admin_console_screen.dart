import 'package:flutter/material.dart';

import '../models/admin_report.dart';
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
  final _maxKeywordsController = TextEditingController();
  final _hookDurationController = TextEditingController();
  final _scanIntervalController = TextEditingController();
  final _youtubeRegionController = TextEditingController();
  final _googleRegionController = TextEditingController();
  final _tiktokRegionController = TextEditingController();

  AdminOverviewReport? _overview;
  RecommendationAdminReport? _recommendationReport;
  List<ProfileComparison> _comparisons = const [];
  AdminSettings? _settings;
  String? _error;
  bool _loading = false;
  bool _savingSettings = false;
  bool _enableYoutube = true;
  bool _enableGoogle = true;
  bool _enableTiktok = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _maxKeywordsController.dispose();
    _hookDurationController.dispose();
    _scanIntervalController.dispose();
    _youtubeRegionController.dispose();
    _googleRegionController.dispose();
    _tiktokRegionController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final overview = await _repository.getOverviewReport();
      final recommendationReport = await _repository.getRecommendationReport();
      final comparisons = await _repository.compareProfiles();
      final settings = await _repository.getSettings();
      if (!mounted) return;
      setState(() {
        _overview = overview;
        _recommendationReport = recommendationReport;
        _comparisons = comparisons;
        _settings = settings;
        _maxKeywordsController.text = '${settings.maxKeywordsDisplay}';
        _hookDurationController.text = '${settings.hookAnalysisDuration}';
        _scanIntervalController.text = '${settings.autoScanIntervalHours}';
        _youtubeRegionController.text = settings.youtubeRegion;
        _googleRegionController.text = settings.googleRegion;
        _tiktokRegionController.text = settings.tiktokRegion;
        _enableYoutube = settings.enableYoutubeTrending;
        _enableGoogle = settings.enableGoogleTrends;
        _enableTiktok = settings.enableTiktokTrending;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _saveSettings() async {
    setState(() {
      _savingSettings = true;
      _error = null;
    });
    try {
      final settings = await _repository.updateSettings({
        'max_keywords_display': int.tryParse(_maxKeywordsController.text.trim()) ?? 10,
        'hook_analysis_duration': int.tryParse(_hookDurationController.text.trim()) ?? 60,
        'auto_scan_interval_hours': int.tryParse(_scanIntervalController.text.trim()) ?? 6,
        'youtube_region': _youtubeRegionController.text.trim().toUpperCase(),
        'google_region': _googleRegionController.text.trim().toUpperCase(),
        'tiktok_region': _tiktokRegionController.text.trim().toUpperCase(),
        'enable_youtube_trending': _enableYoutube,
        'enable_google_trends': _enableGoogle,
        'enable_tiktok_trending': _enableTiktok,
      });
      if (!mounted) return;
      setState(() => _settings = settings);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Admin settings saved')),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _savingSettings = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final overview = _overview;
    final report = _recommendationReport;
    final datasetHealth = report?.datasetHealth;
    final profileHealth = report?.profileHealth;

    return AppShell(
      title: 'Admin Console',
      currentRoute: '/admin-console',
      isAdmin: true,
      actions: [IconButton(onPressed: _load, icon: const Icon(Icons.refresh))],
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : overview == null || report == null
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
                          _AdminMetricCard(title: 'Datasets', value: '${overview.datasetTotal}'),
                          _AdminMetricCard(title: 'Cluster Runs', value: '${overview.clusterRunTotal}'),
                          _AdminMetricCard(title: 'Logs', value: '${overview.systemLogTotal}'),
                          _AdminMetricCard(title: 'YouTube Rows', value: '${datasetHealth?.youtubeDatasetContents ?? 0}'),
                          _AdminMetricCard(title: 'Google Rows', value: '${datasetHealth?.googleDatasetContents ?? 0}'),
                          _AdminMetricCard(title: 'TikTok Rows', value: '${datasetHealth?.tiktokDatasetContents ?? 0}'),
                        ],
                      ),
                      const SizedBox(height: 16),
                      _ManagementCard(),
                      const SizedBox(height: 16),
                      _SettingsPanel(
                        hasSettings: _settings != null,
                        maxKeywordsController: _maxKeywordsController,
                        hookDurationController: _hookDurationController,
                        scanIntervalController: _scanIntervalController,
                        youtubeRegionController: _youtubeRegionController,
                        googleRegionController: _googleRegionController,
                        tiktokRegionController: _tiktokRegionController,
                        enableYoutube: _enableYoutube,
                        enableGoogle: _enableGoogle,
                        enableTiktok: _enableTiktok,
                        saving: _savingSettings,
                        onEnableYoutubeChanged: (value) => setState(() => _enableYoutube = value),
                        onEnableGoogleChanged: (value) => setState(() => _enableGoogle = value),
                        onEnableTiktokChanged: (value) => setState(() => _enableTiktok = value),
                        onSave: _saveSettings,
                      ),
                      const SizedBox(height: 16),
                      _RecommendationCoreCard(
                        report: report,
                        profileHealth: profileHealth,
                        datasetHealth: datasetHealth,
                      ),
                      const SizedBox(height: 16),
                      _VisualSummaryCard(overview: overview),
                      const SizedBox(height: 16),
                      _ComparisonCard(comparisons: _comparisons),
                    ],
                  ),
                ),
    );
  }
}

class _ManagementCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        children: [
          ListTile(
            leading: const Icon(Icons.storage_outlined),
            title: const Text('Datasets'),
            subtitle: const Text('View, add, and update YouTube, Google, and TikTok records'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.pushNamed(context, '/admin-datasets'),
          ),
          ListTile(
            leading: const Icon(Icons.bubble_chart_outlined),
            title: const Text('Cluster Runs'),
            subtitle: const Text('Run and inspect KMeans/HDBSCAN cluster jobs'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.pushNamed(context, '/admin-clusters'),
          ),
          ListTile(
            leading: const Icon(Icons.receipt_long_outlined),
            title: const Text('System Logs'),
            subtitle: const Text('Review success and failure logs'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.pushNamed(context, '/admin-logs'),
          ),
        ],
      ),
    );
  }
}

class _SettingsPanel extends StatelessWidget {
  const _SettingsPanel({
    required this.hasSettings,
    required this.maxKeywordsController,
    required this.hookDurationController,
    required this.scanIntervalController,
    required this.youtubeRegionController,
    required this.googleRegionController,
    required this.tiktokRegionController,
    required this.enableYoutube,
    required this.enableGoogle,
    required this.enableTiktok,
    required this.saving,
    required this.onEnableYoutubeChanged,
    required this.onEnableGoogleChanged,
    required this.onEnableTiktokChanged,
    required this.onSave,
  });

  final bool hasSettings;
  final TextEditingController maxKeywordsController;
  final TextEditingController hookDurationController;
  final TextEditingController scanIntervalController;
  final TextEditingController youtubeRegionController;
  final TextEditingController googleRegionController;
  final TextEditingController tiktokRegionController;
  final bool enableYoutube;
  final bool enableGoogle;
  final bool enableTiktok;
  final bool saving;
  final ValueChanged<bool> onEnableYoutubeChanged;
  final ValueChanged<bool> onEnableGoogleChanged;
  final ValueChanged<bool> onEnableTiktokChanged;
  final VoidCallback onSave;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Analysis Settings', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(child: TextField(controller: maxKeywordsController, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Max keywords'))),
                const SizedBox(width: 10),
                Expanded(child: TextField(controller: hookDurationController, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Hook duration sec'))),
                const SizedBox(width: 10),
                Expanded(child: TextField(controller: scanIntervalController, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Scan interval hr'))),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(child: TextField(controller: youtubeRegionController, decoration: const InputDecoration(labelText: 'YouTube region'))),
                const SizedBox(width: 10),
                Expanded(child: TextField(controller: googleRegionController, decoration: const InputDecoration(labelText: 'Google region'))),
                const SizedBox(width: 10),
                Expanded(child: TextField(controller: tiktokRegionController, decoration: const InputDecoration(labelText: 'TikTok region'))),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 12,
              children: [
                FilterChip(label: const Text('YouTube'), selected: enableYoutube, onSelected: onEnableYoutubeChanged),
                FilterChip(label: const Text('Google'), selected: enableGoogle, onSelected: onEnableGoogleChanged),
                FilterChip(label: const Text('TikTok'), selected: enableTiktok, onSelected: onEnableTiktokChanged),
              ],
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: !hasSettings || saving ? null : onSave,
              icon: saving
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save_outlined),
              label: Text(saving ? 'Saving...' : 'Save settings'),
            ),
          ],
        ),
      ),
    );
  }
}

class _RecommendationCoreCard extends StatelessWidget {
  const _RecommendationCoreCard({
    required this.report,
    required this.profileHealth,
    required this.datasetHealth,
  });

  final RecommendationAdminReport report;
  final ProfileHealth? profileHealth;
  final DatasetHealth? datasetHealth;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Recommendation Core', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(
              'Datasets ${datasetHealth?.totalDatasetContents ?? 0} | duration coverage ${(datasetHealth?.durationCoverageRatio ?? 0).toStringAsFixed(2)}',
            ),
            const SizedBox(height: 12),
            _DomainBlock(title: 'YouTube domains', domains: profileHealth?.youtubeDomains ?? const []),
            _DomainBlock(title: 'Google domains', domains: profileHealth?.googleDomains ?? const []),
            _DomainBlock(title: 'TikTok domains', domains: profileHealth?.tiktokDomains ?? const []),
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _ProfilePreview(title: 'YouTube Profiles', profiles: report.youtubeProfiles)),
                const SizedBox(width: 12),
                Expanded(child: _ProfilePreview(title: 'Google Profiles', profiles: report.googleProfiles)),
                const SizedBox(width: 12),
                Expanded(child: _ProfilePreview(title: 'TikTok Profiles', profiles: report.tiktokProfiles)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _VisualSummaryCard extends StatelessWidget {
  const _VisualSummaryCard({required this.overview});

  final AdminOverviewReport overview;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Visual Summary', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            const Text('Top Sources'),
            SimpleBarChart(items: overview.topSources),
            const SizedBox(height: 12),
            const Text('Top Categories'),
            SimpleBarChart(items: overview.topCategories),
            const SizedBox(height: 12),
            const Text('Log Status Breakdown'),
            SimpleBarChart(items: overview.statusBreakdown),
          ],
        ),
      ),
    );
  }
}

class _ComparisonCard extends StatelessWidget {
  const _ComparisonCard({required this.comparisons});

  final List<ProfileComparison> comparisons;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('YouTube / Google / TikTok Comparison', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            if (comparisons.isEmpty)
              const Text('No comparable profiles yet')
            else
              ...comparisons.take(9).map((item) => Padding(
                    padding: const EdgeInsets.only(bottom: 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(item.domain, style: const TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 4),
                        Text('${_formatSource(item.leftSource)} ${item.leftSampleSize} | ${_formatSource(item.rightSource)} ${item.rightSampleSize}'),
                        Text('${_formatSource(item.leftSource)} ${item.leftDuration.recommendedRange} | ${_formatSource(item.rightSource)} ${item.rightDuration.recommendedRange}'),
                        const SizedBox(height: 6),
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(child: _KeywordPreview(title: '${_formatSource(item.leftSource)} keywords', keywords: item.leftTopKeywords)),
                            const SizedBox(width: 12),
                            Expanded(child: _KeywordPreview(title: '${_formatSource(item.rightSource)} keywords', keywords: item.rightTopKeywords)),
                          ],
                        ),
                      ],
                    ),
                  )),
          ],
        ),
      ),
    );
  }
}

class _DomainBlock extends StatelessWidget {
  const _DomainBlock({required this.title, required this.domains});

  final String title;
  final List<String> domains;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title),
          const SizedBox(height: 6),
          if (domains.isEmpty)
            const Text('No domains yet')
          else
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: domains.map((domain) => Chip(label: Text(domain))).toList(),
            ),
        ],
      ),
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
          ...visible.map((profile) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text('${profile.domain} | samples ${profile.sampleSize} | ${profile.duration.recommendedRange}'),
              )),
      ],
    );
  }
}

class _KeywordPreview extends StatelessWidget {
  const _KeywordPreview({required this.title, required this.keywords});

  final String title;
  final List<dynamic> keywords;

  @override
  Widget build(BuildContext context) {
    final labels = keywords.take(4).map((item) => item.keyword.toString()).toList();
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

String _formatSource(String value) {
  final normalized = value.trim().toLowerCase();
  if (normalized == 'youtube') return 'YouTube';
  if (normalized == 'google') return 'Google';
  if (normalized == 'tiktok') return 'TikTok';
  return value;
}
