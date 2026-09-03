import 'package:flutter/material.dart';

import '../models/admin_report.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminConsoleScreen extends StatefulWidget {
  const AdminConsoleScreen({super.key, this.repository});

  final AdminRepository? repository;

  @override
  State<AdminConsoleScreen> createState() => _AdminConsoleScreenState();
}

class _AdminConsoleScreenState extends State<AdminConsoleScreen> {
  late final AdminRepository _repository;
  final _hookDurationController = TextEditingController();

  RecommendationAdminReport? _recommendationReport;
  AdminSettings? _settings;
  String? _error;
  bool _loading = false;
  bool _savingSettings = false;

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? AdminRepository();
    _load();
  }

  @override
  void dispose() {
    _hookDurationController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final recommendationReport = await _repository.getRecommendationReport();
      final settings = await _repository.getSettings();
      if (!mounted) return;
      setState(() {
        _recommendationReport = recommendationReport;
        _settings = settings;
        _hookDurationController.text = '${settings.hookAnalysisDuration}';
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
        'hook_analysis_duration':
            int.tryParse(_hookDurationController.text.trim()) ?? 60,
      });
      if (!mounted) return;
      setState(() => _settings = settings);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('บันทึกการตั้งค่าแล้ว')),
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
          : report == null
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
                              title: 'ข้อมูลพร้อมใช้กับโมเดล',
                              value:
                                  '${datasetHealth?.totalDatasetContents ?? 0}'),
                          _AdminMetricCard(
                              title: 'หมวดที่พร้อมสร้างคำแนะนำ',
                              value: '${profileHealth?.youtubeProfiles ?? 0}'),
                          _AdminMetricCard(
                              title: 'ข้อมูลความยาวพร้อมใช้',
                              value:
                                  '${datasetHealth?.durationCoverageCount ?? 0}'),
                        ],
                      ),
                      const SizedBox(height: 16),
                      _ManagementCard(),
                      const SizedBox(height: 16),
                      _SettingsPanel(
                        hasSettings: _settings != null,
                        hookDurationController: _hookDurationController,
                        saving: _savingSettings,
                        onSave: _saveSettings,
                      ),
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
            leading: const Icon(Icons.fact_check_outlined),
            title: const Text('Dataset Review'),
            subtitle: const Text(
              'Approve, reject, or relabel public YouTube transcript candidates',
            ),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.pushNamed(context, '/admin-dataset-review'),
          ),
          ListTile(
            leading: const Icon(Icons.storage_outlined),
            title: const Text('Datasets'),
            subtitle: const Text('View and update approved dataset records'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => Navigator.pushNamed(context, '/admin-datasets'),
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
    required this.hookDurationController,
    required this.saving,
    required this.onSave,
  });

  final bool hasSettings;
  final TextEditingController hookDurationController;
  final bool saving;
  final VoidCallback onSave;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('การตั้งค่าการวิเคราะห์คลิป',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 360),
              child: TextField(
                controller: hookDurationController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'ช่วงเปิดคลิปที่ใช้วิเคราะห์ (วินาที)',
                ),
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: !hasSettings || saving ? null : onSave,
              icon: saving
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.save_outlined),
              label: Text(saving ? 'กำลังบันทึก...' : 'บันทึกการตั้งค่า'),
            ),
          ],
        ),
      ),
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
