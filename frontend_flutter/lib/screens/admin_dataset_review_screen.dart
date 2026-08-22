import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/dataset_review.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminDatasetReviewScreen extends StatefulWidget {
  const AdminDatasetReviewScreen({super.key, this.repository});

  final AdminRepository? repository;

  @override
  State<AdminDatasetReviewScreen> createState() =>
      _AdminDatasetReviewScreenState();
}

class _AdminDatasetReviewScreenState extends State<AdminDatasetReviewScreen> {
  late final AdminRepository _repository;
  final _searchController = TextEditingController();
  DatasetReviewQueueResult? _queue;
  String _status = 'pending';
  String _leafKey = 'all';
  int? _runId;
  int _offset = 0;
  final int _limit = 12;
  bool _loading = false;
  String? _reviewingId;
  String? _error;

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? AdminRepository();
    _load();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final result = await _repository.listDatasetReviewQueue(
        limit: _limit,
        offset: _offset,
        status: _status,
        leafKey: _leafKey,
        collectionRunId: _runId,
        search: _searchController.text,
      );
      if (!mounted) return;
      setState(() => _queue = result);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _applyFilters() {
    setState(() => _offset = 0);
    _load();
  }

  Future<void> _openVideo(DatasetReviewCandidate candidate) async {
    final uri = Uri.tryParse(candidate.videoUrl);
    if (uri == null ||
        !await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not open the YouTube video')),
      );
    }
  }

  Future<void> _review(
    DatasetReviewCandidate candidate,
    String decision,
  ) async {
    final queue = _queue;
    if (queue == null) return;
    final result = await showDialog<_ReviewDecision>(
      context: context,
      builder: (context) => _ReviewDialog(
        candidate: candidate,
        taxonomy: queue.taxonomy,
        initialDecision: decision,
      ),
    );
    if (result == null || !mounted) return;

    setState(() {
      _reviewingId = candidate.youtubeId;
      _error = null;
    });
    try {
      await _repository.reviewDatasetCandidate(
        candidate: candidate,
        decision: result.decision,
        reviewedLeafKey: result.reviewedLeafKey,
        transcriptQuality: result.transcriptQuality,
        notes: result.notes,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            result.decision == 'approve'
                ? 'Candidate approved and imported into the production dataset'
                : 'Candidate rejected and recorded in the audit log',
          ),
        ),
      );
      await _load();
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _reviewingId = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final queue = _queue;
    return AppShell(
      title: 'Dataset Review',
      currentRoute: '/admin-dataset-review',
      isAdmin: true,
      actions: [
        IconButton(
          onPressed: _loading ? null : _load,
          icon: const Icon(Icons.refresh),
          tooltip: 'Refresh review queue',
        ),
      ],
      child: _error != null && queue == null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : RefreshIndicator(
              onRefresh: _load,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  if (_loading) const LinearProgressIndicator(),
                  if (_error != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: MaterialBanner(
                        content: Text(_error!),
                        actions: [
                          TextButton(
                            onPressed: () => setState(() => _error = null),
                            child: const Text('Dismiss'),
                          ),
                        ],
                      ),
                    ),
                  if (queue != null) ...[
                    _ReviewSummaryBand(summary: queue.summary),
                    const SizedBox(height: 16),
                    _CollectionRunProgress(
                      runs: queue.runs,
                      taxonomy: queue.taxonomy,
                    ),
                    const SizedBox(height: 16),
                    _ReviewFilters(
                      searchController: _searchController,
                      status: _status,
                      leafKey: _leafKey,
                      runId: _runId,
                      taxonomy: queue.taxonomy,
                      runs: queue.runs,
                      onStatusChanged: (value) {
                        setState(() => _status = value);
                        _applyFilters();
                      },
                      onLeafChanged: (value) {
                        setState(() => _leafKey = value);
                        _applyFilters();
                      },
                      onRunChanged: (value) {
                        setState(() => _runId = value);
                        _applyFilters();
                      },
                      onSearch: _applyFilters,
                    ),
                    const SizedBox(height: 16),
                    _TaxonomyProgress(taxonomy: queue.taxonomy),
                    const SizedBox(height: 16),
                    if (queue.items.isEmpty)
                      EmptyStateView(
                        title: _status == 'pending'
                            ? 'No candidates waiting for review'
                            : 'No review records found',
                        message:
                            'Collect more YouTube CC candidates or adjust the filters.',
                        icon: Icons.fact_check_outlined,
                      )
                    else
                      ...queue.items.map(
                        (candidate) => _CandidateReviewCard(
                          candidate: candidate,
                          reviewing: _reviewingId == candidate.youtubeId,
                          onOpenVideo: () => _openVideo(candidate),
                          onApprove: () => _review(candidate, 'approve'),
                          onReject: () => _review(candidate, 'reject'),
                        ),
                      ),
                    PaginationBar(
                      offset: _offset,
                      limit: _limit,
                      total: queue.total,
                      onPrevious: _offset <= 0
                          ? null
                          : () {
                              setState(() {
                                _offset = (_offset - _limit).clamp(0, _offset);
                              });
                              _load();
                            },
                      onNext: _offset + _limit >= queue.total
                          ? null
                          : () {
                              setState(() => _offset += _limit);
                              _load();
                            },
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}

class _ReviewSummaryBand extends StatelessWidget {
  const _ReviewSummaryBand({required this.summary});

  final DatasetReviewSummary summary;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Wrap(
        spacing: 28,
        runSpacing: 12,
        children: [
          _SummaryMetric(label: 'All candidates', value: summary.total),
          _SummaryMetric(label: 'Pending', value: summary.pending),
          _SummaryMetric(label: 'Approved', value: summary.approved),
          _SummaryMetric(label: 'Rejected', value: summary.rejected),
        ],
      ),
    );
  }
}

class _SummaryMetric extends StatelessWidget {
  const _SummaryMetric({required this.label, required this.value});

  final String label;
  final int value;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 130,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$value', style: Theme.of(context).textTheme.headlineSmall),
          Text(label, style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}

class _CollectionRunProgress extends StatelessWidget {
  const _CollectionRunProgress({
    required this.runs,
    required this.taxonomy,
  });

  final List<DatasetReviewRun> runs;
  final List<DatasetReviewTaxonomyLeaf> taxonomy;

  String _leafLabel(String key) {
    for (final leaf in taxonomy) {
      if (leaf.leafKey == key) return leaf.level3;
    }
    return key;
  }

  String _statusLabel(String status) {
    return switch (status) {
      'quota_waiting' => 'Waiting for YouTube quota',
      'running' => 'Collecting',
      'collected' => 'Collection complete',
      'partial' => 'Partial collection',
      'review_pending' => 'Waiting for review',
      'partially_reviewed' => 'Partially reviewed',
      'reviewed' => 'Review complete',
      'failed' => 'Collection failed',
      _ => status.replaceAll('_', ' '),
    };
  }

  IconData _statusIcon(String status) {
    return switch (status) {
      'quota_waiting' => Icons.schedule_outlined,
      'running' => Icons.sync,
      'collected' || 'reviewed' => Icons.check_circle_outline,
      'failed' => Icons.error_outline,
      _ => Icons.data_usage_outlined,
    };
  }

  Color? _statusColor(BuildContext context, String status) {
    return switch (status) {
      'quota_waiting' => Colors.orange.shade800,
      'running' => Theme.of(context).colorScheme.primary,
      'collected' || 'reviewed' => Colors.green.shade700,
      'failed' => Theme.of(context).colorScheme.error,
      _ => null,
    };
  }

  @override
  Widget build(BuildContext context) {
    if (runs.isEmpty) return const SizedBox.shrink();
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 6),
            child: Text(
              'Collection progress',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          ...runs.map((run) {
            final progress = run.progress;
            final languageCounts = progress.languageCounts;
            final statusColor = _statusColor(context, run.status);
            return ExpansionTile(
              initiallyExpanded:
                  run.status == 'running' || run.status == 'quota_waiting',
              leading: Icon(
                _statusIcon(run.status),
                color: statusColor,
              ),
              title: Text(
                'Run ${run.collectionRunId} · ${_statusLabel(run.status)}',
              ),
              subtitle: Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${progress.acceptedTotal}/${progress.targetTotal} candidates'
                      ' · TH ${languageCounts['th'] ?? 0}'
                      ' · EN ${languageCounts['en'] ?? 0}'
                      ' · ${progress.uniqueChannels} channels',
                    ),
                    const SizedBox(height: 6),
                    LinearProgressIndicator(
                      value: progress.targetTotal <= 0
                          ? 0
                          : (progress.percent / 100).clamp(0, 1),
                    ),
                  ],
                ),
              ),
              children: [
                if (run.status == 'quota_waiting')
                  const ListTile(
                    dense: true,
                    leading: Icon(Icons.info_outline),
                    title: Text(
                      'Checkpoint saved. Resume this run after the YouTube quota resets.',
                    ),
                  ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
                  child: Wrap(
                    spacing: 24,
                    runSpacing: 16,
                    children: progress.byLeaf.map((leaf) {
                      final thai = leaf.languageCounts['th'] ?? 0;
                      final english = leaf.languageCounts['en'] ?? 0;
                      return SizedBox(
                        width: 270,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '${_leafLabel(leaf.leafKey)} '
                              '${leaf.accepted}/${leaf.target}',
                              style: Theme.of(context).textTheme.labelLarge,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'TH $thai/${leaf.thaiMinimum} · EN $english · '
                              '${leaf.uniqueChannels} channels · '
                              'max ${leaf.maxVideosPerChannel}/channel',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                            const SizedBox(height: 6),
                            LinearProgressIndicator(
                              value: leaf.target <= 0
                                  ? 0
                                  : (leaf.percent / 100).clamp(0, 1),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ),
                ),
              ],
            );
          }),
        ],
      ),
    );
  }
}

class _ReviewFilters extends StatelessWidget {
  const _ReviewFilters({
    required this.searchController,
    required this.status,
    required this.leafKey,
    required this.runId,
    required this.taxonomy,
    required this.runs,
    required this.onStatusChanged,
    required this.onLeafChanged,
    required this.onRunChanged,
    required this.onSearch,
  });

  final TextEditingController searchController;
  final String status;
  final String leafKey;
  final int? runId;
  final List<DatasetReviewTaxonomyLeaf> taxonomy;
  final List<DatasetReviewRun> runs;
  final ValueChanged<String> onStatusChanged;
  final ValueChanged<String> onLeafChanged;
  final ValueChanged<int?> onRunChanged;
  final VoidCallback onSearch;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final fieldWidth = constraints.maxWidth < 720
            ? constraints.maxWidth
            : (constraints.maxWidth - 36) / 4;
        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            SizedBox(
              width: fieldWidth,
              child: TextField(
                controller: searchController,
                decoration: const InputDecoration(
                  labelText: 'Search title, channel, or transcript',
                  prefixIcon: Icon(Icons.search),
                ),
                onSubmitted: (_) => onSearch(),
              ),
            ),
            SizedBox(
              width: fieldWidth,
              child: DropdownButtonFormField<String>(
                initialValue: status,
                decoration: const InputDecoration(labelText: 'Review status'),
                items: const [
                  DropdownMenuItem(value: 'pending', child: Text('Pending')),
                  DropdownMenuItem(value: 'approved', child: Text('Approved')),
                  DropdownMenuItem(value: 'rejected', child: Text('Rejected')),
                  DropdownMenuItem(value: 'all', child: Text('All')),
                ],
                onChanged: (value) {
                  if (value != null) onStatusChanged(value);
                },
              ),
            ),
            SizedBox(
              width: fieldWidth,
              child: DropdownButtonFormField<String>(
                initialValue: leafKey,
                isExpanded: true,
                decoration: const InputDecoration(labelText: 'Category'),
                items: [
                  const DropdownMenuItem(
                      value: 'all', child: Text('All categories')),
                  ...taxonomy.map(
                    (leaf) => DropdownMenuItem(
                      value: leaf.leafKey,
                      child: Text(leaf.level3, overflow: TextOverflow.ellipsis),
                    ),
                  ),
                ],
                onChanged: (value) {
                  if (value != null) onLeafChanged(value);
                },
              ),
            ),
            SizedBox(
              width: fieldWidth,
              child: DropdownButtonFormField<int?>(
                initialValue: runId,
                isExpanded: true,
                decoration: const InputDecoration(labelText: 'Collection run'),
                items: [
                  const DropdownMenuItem<int?>(
                      value: null, child: Text('All runs')),
                  ...runs.map(
                    (run) => DropdownMenuItem<int?>(
                      value: run.collectionRunId,
                      child: Text(
                        'Run ${run.collectionRunId} (${run.pending} pending)',
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                ],
                onChanged: onRunChanged,
              ),
            ),
          ],
        );
      },
    );
  }
}

class _TaxonomyProgress extends StatelessWidget {
  const _TaxonomyProgress({required this.taxonomy});

  final List<DatasetReviewTaxonomyLeaf> taxonomy;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Verified coverage',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: taxonomy
              .map(
                (leaf) => Chip(
                  avatar: Icon(
                    leaf.ready ? Icons.check_circle : Icons.pending_outlined,
                    size: 18,
                    color: leaf.ready ? Colors.green : null,
                  ),
                  label: Text(
                    '${leaf.level3} ${leaf.verifiedSampleCount}/${leaf.minimumSampleCount}',
                  ),
                ),
              )
              .toList(),
        ),
      ],
    );
  }
}

class _CandidateReviewCard extends StatelessWidget {
  const _CandidateReviewCard({
    required this.candidate,
    required this.reviewing,
    required this.onOpenVideo,
    required this.onApprove,
    required this.onReject,
  });

  final DatasetReviewCandidate candidate;
  final bool reviewing;
  final VoidCallback onOpenVideo;
  final VoidCallback onApprove;
  final VoidCallback onReject;

  @override
  Widget build(BuildContext context) {
    final passedChecks =
        candidate.automatedChecks.values.where((value) => value).length;
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: Theme.of(context).dividerColor),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Text(
                    candidate.title,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                const SizedBox(width: 8),
                _StatusChip(status: candidate.reviewStatus),
                IconButton(
                  onPressed: onOpenVideo,
                  icon: const Icon(Icons.open_in_new),
                  tooltip: 'Open video on YouTube',
                ),
              ],
            ),
            Text(
              '${candidate.channelTitle} | ${candidate.durationSeconds}s | '
              '${candidate.transcriptLanguage.toUpperCase()} ${candidate.captionType} captions',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Chip(label: Text('Suggested: ${candidate.proposedLeafKey}')),
                Chip(
                  avatar: Icon(
                    candidate.allAutomatedChecksPass
                        ? Icons.verified_outlined
                        : Icons.warning_amber_outlined,
                    size: 18,
                  ),
                  label: Text(
                    'Automated checks $passedChecks/${candidate.automatedChecks.length}',
                  ),
                ),
                const Chip(
                  avatar: Icon(Icons.model_training_outlined, size: 18),
                  label: Text('Classification + keywords'),
                ),
                if (candidate.transcriptAcquisitionMethod ==
                    'notebooklm_manual_source')
                  const Chip(
                    avatar: Icon(Icons.text_snippet_outlined, size: 18),
                    label: Text('NotebookLM source'),
                  ),
                if (candidate.transcriptScope == 'full_video')
                  const Chip(
                    avatar: Icon(Icons.subject_outlined, size: 18),
                    label: Text('Full video transcript'),
                  ),
                Chip(
                  avatar: Icon(
                    candidate.datasetUsage['duration_recommendation'] == true
                        ? Icons.schedule_outlined
                        : Icons.timer_off_outlined,
                    size: 18,
                  ),
                  label: Text(
                    candidate.datasetUsage['duration_recommendation'] == true
                        ? 'Duration evidence'
                        : 'Excluded from duration evidence',
                  ),
                ),
                ...candidate.evidenceTerms.take(6).map(
                      (term) => Chip(label: Text(term)),
                    ),
              ],
            ),
            const SizedBox(height: 10),
            SelectionArea(
              child: Text(
                candidate.transcriptPreview,
                maxLines: 5,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              childrenPadding: const EdgeInsets.only(bottom: 8),
              title: Text(
                'Full transcript (${candidate.transcript.length} characters)',
                style: Theme.of(context).textTheme.labelLarge,
              ),
              children: [
                ConstrainedBox(
                  constraints: const BoxConstraints(maxHeight: 320),
                  child: SingleChildScrollView(
                    child: SelectionArea(
                      child: Text(candidate.transcript),
                    ),
                  ),
                ),
              ],
            ),
            if (candidate.reviewer != null) ...[
              const SizedBox(height: 10),
              Text(
                'Reviewed by ${candidate.reviewer}'
                '${candidate.reviewedLeafKey == null ? '' : ' as ${candidate.reviewedLeafKey}'}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            const SizedBox(height: 14),
            if (reviewing)
              const LinearProgressIndicator()
            else
              Wrap(
                spacing: 10,
                runSpacing: 8,
                children: [
                  FilledButton.icon(
                    onPressed: onApprove,
                    icon: const Icon(Icons.check),
                    label: const Text('Approve'),
                  ),
                  OutlinedButton.icon(
                    onPressed: onReject,
                    icon: const Icon(Icons.close),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Theme.of(context).colorScheme.error,
                      side: BorderSide(
                          color: Theme.of(context).colorScheme.error),
                    ),
                    label: const Text('Reject'),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.status});

  final String status;

  @override
  Widget build(BuildContext context) {
    final color = switch (status) {
      'approved' => Colors.green,
      'rejected' => Theme.of(context).colorScheme.error,
      _ => Colors.orange,
    };
    return Chip(
      avatar: Icon(
        status == 'approved'
            ? Icons.check_circle_outline
            : status == 'rejected'
                ? Icons.cancel_outlined
                : Icons.pending_outlined,
        size: 18,
        color: color,
      ),
      label: Text(status),
    );
  }
}

class _ReviewDecision {
  const _ReviewDecision({
    required this.decision,
    required this.notes,
    this.reviewedLeafKey,
    this.transcriptQuality,
  });

  final String decision;
  final String? reviewedLeafKey;
  final String? transcriptQuality;
  final String notes;
}

class _ReviewDialog extends StatefulWidget {
  const _ReviewDialog({
    required this.candidate,
    required this.taxonomy,
    required this.initialDecision,
  });

  final DatasetReviewCandidate candidate;
  final List<DatasetReviewTaxonomyLeaf> taxonomy;
  final String initialDecision;

  @override
  State<_ReviewDialog> createState() => _ReviewDialogState();
}

class _ReviewDialogState extends State<_ReviewDialog> {
  late String _decision;
  late String _leafKey;
  String _quality = 'good';
  final _notesController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _decision = widget.initialDecision;
    _leafKey =
        widget.candidate.reviewedLeafKey ?? widget.candidate.proposedLeafKey;
    _quality = widget.candidate.transcriptQuality ?? 'good';
    _notesController.text = widget.candidate.reviewNotes ?? '';
  }

  @override
  void dispose() {
    _notesController.dispose();
    super.dispose();
  }

  void _submit() {
    Navigator.pop(
      context,
      _ReviewDecision(
        decision: _decision,
        reviewedLeafKey: _decision == 'approve' ? _leafKey : null,
        transcriptQuality: _decision == 'approve' ? _quality : null,
        notes: _notesController.text,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final approve = _decision == 'approve';
    return AlertDialog(
      title: Text(approve ? 'Approve candidate' : 'Reject candidate'),
      content: SizedBox(
        width: 760,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(widget.candidate.title,
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 4),
              Text(
                '${widget.candidate.channelTitle} | '
                '${widget.candidate.durationSeconds}s | '
                '${widget.candidate.transcriptLanguage.toUpperCase()}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 14),
              Text('Transcript', style: Theme.of(context).textTheme.labelLarge),
              const SizedBox(height: 6),
              Container(
                constraints: const BoxConstraints(maxHeight: 240),
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Theme.of(context).dividerColor),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: SelectionArea(
                  child: SingleChildScrollView(
                    child: Text(widget.candidate.transcript),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(
                    value: 'approve',
                    icon: Icon(Icons.check),
                    label: Text('Approve'),
                  ),
                  ButtonSegment(
                    value: 'reject',
                    icon: Icon(Icons.close),
                    label: Text('Reject'),
                  ),
                ],
                selected: {_decision},
                onSelectionChanged: (values) {
                  setState(() => _decision = values.first);
                },
              ),
              if (approve) ...[
                const SizedBox(height: 14),
                DropdownButtonFormField<String>(
                  initialValue: _leafKey,
                  isExpanded: true,
                  decoration:
                      const InputDecoration(labelText: 'Verified category'),
                  items: widget.taxonomy
                      .map(
                        (leaf) => DropdownMenuItem(
                          value: leaf.leafKey,
                          child: Text(
                            leaf.path,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      )
                      .toList(),
                  onChanged: (value) {
                    if (value != null) setState(() => _leafKey = value);
                  },
                ),
                const SizedBox(height: 12),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(
                        value: 'good', label: Text('Good transcript')),
                    ButtonSegment(
                      value: 'acceptable',
                      label: Text('Acceptable'),
                    ),
                  ],
                  selected: {_quality},
                  onSelectionChanged: (values) {
                    setState(() => _quality = values.first);
                  },
                ),
              ],
              const SizedBox(height: 14),
              TextField(
                controller: _notesController,
                minLines: 2,
                maxLines: 4,
                decoration: InputDecoration(
                  labelText:
                      approve ? 'Review notes (optional)' : 'Reject reason',
                ),
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton.icon(
          onPressed: _submit,
          icon: Icon(approve ? Icons.check : Icons.close),
          style: approve
              ? null
              : FilledButton.styleFrom(
                  backgroundColor: Theme.of(context).colorScheme.error,
                ),
          label: Text(approve ? 'Confirm approval' : 'Confirm rejection'),
        ),
      ],
    );
  }
}
