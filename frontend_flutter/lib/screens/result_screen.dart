import 'package:flutter/material.dart';

import '../models/common_models.dart';
import '../models/recommendation_result.dart';
import '../repositories/content_repository.dart';
import '../state/auth_scope.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class ResultScreenArgs {
  const ResultScreenArgs({this.initialData, this.contentId});

  final AnalysisResultViewData? initialData;
  final int? contentId;
}

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  final _repository = ContentRepository();
  AnalysisResultViewData? _data;
  String? _error;
  bool _initialized = false;
  bool _saveLoading = false;
  bool _saved = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_initialized) return;
    _initialized = true;

    final args =
        ModalRoute.of(context)?.settings.arguments as ResultScreenArgs?;
    if (args?.initialData != null) {
      _data = args!.initialData;
      _saved = _data?.saved ?? false;
    } else if (args?.contentId != null) {
      _loadContent(args!.contentId!);
    }
  }

  Future<void> _loadContent(int contentId) async {
    try {
      final response = await _repository.getContentResult(contentId);
      if (!mounted) return;
      setState(() {
        _data = response;
        _saved = response.saved;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    }
  }

  Future<void> _saveToIdeas() async {
    if (_saved || _data?.contentId != null) {
      setState(() => _saved = true);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('This analysis is already saved in My Ideas.'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    setState(() => _saveLoading = true);
    try {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Use Analyze & Save to store this result in My Ideas.'),
          duration: Duration(seconds: 2),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${e.toString()}')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _saveLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = _data;
    final recommendation = data?.recommendation;
    final classification = recommendation?.classification;
    final classifierConfidence = (classification?.confidence ?? 0) * 100;
    final domain = classification?.displayCategory ??
        recommendation?.domain ??
        data?.fallbackDomain ??
        '-';
    final userKeywords = recommendation?.userKeywords ?? const <String>[];
    final contentKeywords = recommendation?.contentKeywords ?? const <String>[];
    final hookTerms = recommendation?.hookTerms ?? const <String>[];
    final missingKeywords =
        recommendation?.missingKeywords ?? const <KeywordScore>[];
    final hookKeywords = recommendation?.hookKeywords ?? const <KeywordScore>[];
    final duration = recommendation?.duration;
    final evidence = recommendation?.evidence;
    final datasetProfile = recommendation?.datasetProfile;
    final isSaved = _saved || data?.contentId != null;

    return AppShell(
      title: 'Analysis Result',
      isAdmin: AuthScope.of(context).isAdmin,
      actions: [
        if (isSaved)
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 12),
            child: Chip(
              avatar: Icon(Icons.bookmark_added_outlined, size: 18),
              label: Text('Saved'),
            ),
          )
        else if (!_saveLoading)
          IconButton(
            onPressed: _saveToIdeas,
            icon: const Icon(Icons.bookmark_outline),
            tooltip: 'Save to My Ideas',
          )
        else
          const Center(
            child: SizedBox(
              width: 24,
              height: 24,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          ),
      ],
      child: _error != null
          ? ErrorStateView(message: _error!)
          : data == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: () async {
                    final contentId = data.contentId;
                    if (contentId != null) {
                      await _loadContent(contentId);
                    }
                  },
                  child: ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      Text(
                        data.title,
                        style: Theme.of(context).textTheme.headlineSmall,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          Chip(
                            avatar:
                                const Icon(Icons.category_outlined, size: 18),
                            label: Text('Type: $domain'),
                          ),
                          Chip(
                            avatar: const Icon(Icons.check_circle, size: 18),
                            label: Text(
                              '${classifierConfidence.toStringAsFixed(0)}% confidence',
                            ),
                          ),
                          if (isSaved)
                            const Chip(
                              avatar:
                                  Icon(Icons.bookmark_added_outlined, size: 18),
                              label: Text('Saved in My Ideas'),
                            ),
                        ],
                      ),
                      const SizedBox(height: 24),
                      _ScopeSummaryCard(
                        domain: domain,
                        confidence: classifierConfidence,
                        userKeywords: userKeywords,
                        missingKeywords: missingKeywords,
                        duration: duration,
                      ),
                      const SizedBox(height: 24),
                      if (evidence != null && datasetProfile != null) ...[
                        _EvidenceCard(
                          evidence: evidence,
                          datasetProfile: datasetProfile,
                        ),
                        const SizedBox(height: 24),
                      ],
                      if (data.transcript.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Transcript Preview',
                          icon: Icons.subtitles_outlined,
                        ),
                        Card(
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Text(
                              data.transcript,
                              maxLines: 5,
                              overflow: TextOverflow.ellipsis,
                              style: Theme.of(context).textTheme.bodyMedium,
                            ),
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],
                      const _SectionHeader(
                        title: 'Content Type',
                        icon: Icons.account_tree_outlined,
                      ),
                      _ClassificationCard(
                        domain: domain,
                        confidence: classifierConfidence,
                        classification: classification,
                      ),
                      const SizedBox(height: 24),
                      const _SectionHeader(
                        title: 'Keywords Found in Your Clip',
                        icon: Icons.key_outlined,
                      ),
                      _StringKeywordCard(
                        keywords: userKeywords,
                        emptyMessage:
                            'No keywords were detected from this clip.',
                      ),
                      const SizedBox(height: 24),
                      if (contentKeywords.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Content Keywords',
                          icon: Icons.article_outlined,
                        ),
                        _StringKeywordCard(
                          keywords: contentKeywords,
                          emptyMessage: 'No content keywords were detected.',
                        ),
                        const SizedBox(height: 24),
                      ],
                      if (hookTerms.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Hook Terms From Opening Segment',
                          icon: Icons.bolt_outlined,
                        ),
                        _StringKeywordCard(
                          keywords: hookTerms,
                          emptyMessage: 'No hook terms were detected.',
                        ),
                        const SizedBox(height: 24),
                      ],
                      if (missingKeywords.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Keyword Gap From Same-Category Transcripts',
                          icon: Icons.auto_awesome_outlined,
                        ),
                        Text(
                          'These terms appear frequently in verified $domain transcripts but were not found in your clip.',
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: Colors.grey),
                        ),
                        const SizedBox(height: 4),
                        if (evidence != null)
                          Text(
                            evidence.keywordScoreExplanation,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        const SizedBox(height: 8),
                        _ScoredKeywordCard(keywords: missingKeywords),
                        const SizedBox(height: 24),
                      ],
                      if (duration != null) ...[
                        const _SectionHeader(
                          title: 'Recommended Video Duration',
                          icon: Icons.schedule_outlined,
                        ),
                        _DurationCard(duration: duration),
                        if (evidence != null &&
                            evidence.durationExplanation.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Text(
                            evidence.durationExplanation,
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                        const SizedBox(height: 24),
                      ],
                      if (hookKeywords.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'Hook Keywords Recommendation',
                          icon: Icons.lightbulb_outline,
                        ),
                        _ScoredKeywordCard(keywords: hookKeywords),
                        const SizedBox(height: 24),
                      ],
                      Row(
                        children: [
                          Expanded(
                            child: FilledButton.icon(
                              onPressed: isSaved ? null : _saveToIdeas,
                              icon: Icon(isSaved
                                  ? Icons.bookmark_added_outlined
                                  : Icons.bookmark_outline),
                              label: Text(isSaved ? 'Saved' : 'Save Idea'),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: () =>
                                  Navigator.pushNamed(context, '/dashboard'),
                              icon: const Icon(Icons.home),
                              label: const Text('Back to Dashboard'),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),
                ),
    );
  }
}

class _ScopeSummaryCard extends StatelessWidget {
  const _ScopeSummaryCard({
    required this.domain,
    required this.confidence,
    required this.userKeywords,
    required this.missingKeywords,
    required this.duration,
  });

  final String domain;
  final double confidence;
  final List<String> userKeywords;
  final List<KeywordScore> missingKeywords;
  final DurationRecommendation? duration;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _SummaryRow(
              icon: Icons.category_outlined,
              label: 'Clip type',
              value: '$domain (${confidence.toStringAsFixed(0)}%)',
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.key_outlined,
              label: 'Keywords in clip',
              value:
                  userKeywords.isEmpty ? '-' : userKeywords.take(6).join(', '),
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.auto_awesome_outlined,
              label: 'Keyword gap',
              value: missingKeywords.isEmpty
                  ? 'No major gap detected'
                  : missingKeywords
                      .take(6)
                      .map((item) => item.keyword)
                      .join(', '),
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.schedule_outlined,
              label: 'Recommended duration',
              value: duration?.recommendedRange ?? '-',
            ),
          ],
        ),
      ),
    );
  }
}

class _EvidenceCard extends StatelessWidget {
  const _EvidenceCard({
    required this.evidence,
    required this.datasetProfile,
  });

  final RecommendationEvidence evidence;
  final DatasetProfile datasetProfile;

  @override
  Widget build(BuildContext context) {
    final sourceCounts = evidence.sourcePlatformCounts.isNotEmpty
        ? evidence.sourcePlatformCounts
        : datasetProfile.sourcePlatformCounts;
    final sourceText = sourceCounts.isEmpty
        ? evidence.source
        : sourceCounts.entries
            .map((entry) => '${entry.key}: ${entry.value}')
            .join(', ');
    final languageText = evidence.languageCounts.entries
        .map((entry) => '${entry.key}: ${entry.value}')
        .join(', ');
    final verificationText = <String>[
      if (evidence.verificationStatus.isNotEmpty)
        evidence.verificationStatus.replaceAll('_', ' '),
      if (evidence.licenseName.isNotEmpty) evidence.licenseName,
    ].join(' · ');
    final selectionText = evidence.selectionRule ==
            'single_view_metric_cohort_then_top_40_percent'
        ? 'One compatible view-count cohort, then the top 40% in the same category'
        : evidence.selectionRule ==
                'top_40_percent_by_average_views_per_day_and_engagement_rate'
            ? 'Top 40% in the same category by average views/day and engagement rate'
            : evidence.selectionRule.replaceAll('_', ' ');
    final examples = evidence.exemplarTitles.isNotEmpty
        ? evidence.exemplarTitles.take(3).toList()
        : datasetProfile.exemplarTitles.take(3).toList();
    final lineageParts = <String>[
      if (evidence.datasetSources.isNotEmpty)
        evidence.datasetSources.join(', '),
      if (evidence.datasetVersions.isNotEmpty)
        evidence.datasetVersions.join(', '),
    ];
    final lineageText = lineageParts.isEmpty ? '-' : lineageParts.join(' / ');

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.dataset_outlined,
                  color: Theme.of(context).primaryColor,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Recommendation Evidence',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (evidence.warning != null && evidence.warning!.isNotEmpty) ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                color: Theme.of(context).colorScheme.errorContainer,
                child: Text(
                  evidence.warning!,
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onErrorContainer,
                  ),
                ),
              ),
              const SizedBox(height: 12),
            ],
            _SummaryRow(
              icon: Icons.source_outlined,
              label: 'Data source type',
              value: evidence.dataSourceLabel,
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.account_tree_outlined,
              label: 'Dataset source / version',
              value: lineageText,
            ),
            if (evidence.sourceRecordIds.isNotEmpty) ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.fingerprint_outlined,
                label: 'Auditable source records',
                value: '${evidence.sourceRecordIds.length} record IDs',
              ),
            ],
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.subtitles_outlined,
              label: 'Transcript source',
              value: _transcriptSourceText(evidence),
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.hub_outlined,
              label: 'Platform/source counts',
              value: sourceText,
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.video_library_outlined,
              label: 'Same-type sample size',
              value: evidence.eligiblePoolSize > evidence.datasetSampleSize
                  ? '${evidence.datasetSampleSize} high-performing clips from '
                      '${evidence.eligiblePoolSize} eligible clips'
                  : '${evidence.datasetSampleSize} clips',
            ),
            if (evidence.viewMetricVersion.isNotEmpty) ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.rule_outlined,
                label: 'View metric cohort',
                value: _viewMetricEvidenceText(evidence),
              ),
            ],
            if (verificationText.isNotEmpty) ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.verified_outlined,
                label: 'Verification / license',
                value: verificationText,
              ),
            ],
            if (languageText.isNotEmpty) ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.translate_outlined,
                label: 'Transcript languages',
                value: languageText,
              ),
            ],
            if (evidence.selectionRule.isNotEmpty &&
                evidence.selectionRule != 'none') ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.filter_alt_outlined,
                label: 'Evidence selection',
                value: selectionText,
              ),
            ],
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.score_outlined,
              label: 'Keyword score',
              value: evidence.keywordScoreExplanation,
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.schedule_outlined,
              label: 'Duration source',
              value:
                  '${evidence.durationSource}, ${evidence.durationSampleSize} duration samples',
            ),
            if (evidence.durationSamples.isNotEmpty) ...[
              const Divider(height: 20),
              _SummaryRow(
                icon: Icons.timer_outlined,
                label: 'Duration samples used',
                value: evidence.durationSamples
                    .take(12)
                    .map((seconds) => '${seconds}s')
                    .join(', '),
              ),
            ],
            if (examples.isNotEmpty) ...[
              const Divider(height: 20),
              Text(
                'Verified same-category examples',
                style: Theme.of(context).textTheme.labelMedium,
              ),
              const SizedBox(height: 8),
              for (final title in examples)
                Padding(
                  padding: const EdgeInsets.only(bottom: 6),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.trending_up, size: 18),
                      const SizedBox(width: 8),
                      Expanded(child: Text(title)),
                    ],
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}

String _viewMetricEvidenceText(RecommendationEvidence evidence) {
  final label = switch (evidence.viewMetricVersion) {
    'youtube_qualified_view_v1' =>
      'YouTube qualified views, captured before 24 Aug 2026',
    'youtube_play_start_view_v2' =>
      'YouTube play starts, captured from 24 Aug 2026',
    'google_interest_v1' => 'Google search-interest signal',
    'tiktok_public_view_v1' => 'TikTok public views',
    _ => evidence.viewMetricVersion.replaceAll('_', ' '),
  };
  final cohort = evidence.viewMetricCohortSize > 0
      ? '; ${evidence.viewMetricCohortSize} compatible clips'
      : '';
  final excluded = evidence.excludedIncompatibleViewMetricRows > 0
      ? '; ${evidence.excludedIncompatibleViewMetricRows} incompatible rows excluded'
      : '';
  return '$label$cohort$excluded';
}

class _SummaryRow extends StatelessWidget {
  const _SummaryRow({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: Theme.of(context).textTheme.labelMedium),
              const SizedBox(height: 4),
              Text(value, style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        ),
      ],
    );
  }
}

class _ClassificationCard extends StatelessWidget {
  const _ClassificationCard({
    required this.domain,
    required this.confidence,
    required this.classification,
  });

  final String domain;
  final double confidence;
  final ClassificationResult? classification;

  @override
  Widget build(BuildContext context) {
    final candidates =
        classification?.candidates ?? const <ClassificationCandidate>[];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Predicted category',
                        style: Theme.of(context).textTheme.labelMedium,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        domain,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ],
                  ),
                ),
                Column(
                  children: [
                    Text(
                      '${confidence.toStringAsFixed(0)}%',
                      style:
                          Theme.of(context).textTheme.headlineSmall?.copyWith(
                                color: Theme.of(context).primaryColor,
                              ),
                    ),
                    Text(
                      'confidence',
                      style: Theme.of(context).textTheme.labelSmall,
                    ),
                  ],
                ),
              ],
            ),
            if ((classification?.warning ?? '').isNotEmpty) ...[
              const SizedBox(height: 12),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 18,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      classification!.warning,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                ],
              ),
            ],
            if (candidates.isNotEmpty) ...[
              const SizedBox(height: 16),
              const Divider(height: 1),
              const SizedBox(height: 12),
              Text(
                'Other possible categories',
                style: Theme.of(context).textTheme.labelMedium,
              ),
              const SizedBox(height: 8),
              ...candidates.take(3).map((candidate) {
                final value = candidate.score.clamp(0.0, 1.0).toDouble();
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(candidate.domain),
                          Text(
                            '${(candidate.score * 100).toStringAsFixed(0)}%',
                            style: Theme.of(context).textTheme.labelSmall,
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: value,
                          minHeight: 4,
                        ),
                      ),
                    ],
                  ),
                );
              }),
            ],
          ],
        ),
      ),
    );
  }
}

class _StringKeywordCard extends StatelessWidget {
  const _StringKeywordCard({
    required this.keywords,
    required this.emptyMessage,
  });

  final List<String> keywords;
  final String emptyMessage;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: keywords.isEmpty
            ? Text(emptyMessage)
            : Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final keyword in keywords.take(16))
                    Chip(label: Text(keyword)),
                ],
              ),
      ),
    );
  }
}

class _ScoredKeywordCard extends StatelessWidget {
  const _ScoredKeywordCard({required this.keywords});

  final List<KeywordScore> keywords;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            for (final keyword in keywords)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Text(
                        keyword.keyword,
                        style: Theme.of(context).textTheme.labelMedium,
                      ),
                    ),
                    Chip(
                      label: Text(keyword.score.toStringAsFixed(2)),
                      side: const BorderSide(color: Color(0xFFE0E0E0)),
                      backgroundColor: Colors.transparent,
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

class _DurationCard extends StatelessWidget {
  const _DurationCard({required this.duration});

  final DurationRecommendation duration;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              duration.recommendedRange,
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Source',
                      style: Theme.of(context).textTheme.labelSmall,
                    ),
                    Text(
                      duration.source,
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                  ],
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      'Sample size',
                      style: Theme.of(context).textTheme.labelSmall,
                    ),
                    Text(
                      '${duration.sampleSize} videos',
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({
    required this.title,
    required this.icon,
  });

  final String title;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Icon(icon, color: Theme.of(context).primaryColor),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              title,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
          ),
        ],
      ),
    );
  }
}

String _transcriptSourceText(RecommendationEvidence evidence) {
  final source = evidence.transcriptSource;
  final seconds = evidence.hookSecondsAnalyzed;
  final hookSuffix =
      seconds == null ? '' : ' | first ${seconds}s used for hook';
  if (source == 'speech_to_text') {
    final scope = evidence.transcriptScope == 'full_clip' ? ' (full clip)' : '';
    return 'Speech-to-text$scope$hookSuffix';
  }
  if (source == 'fallback_filename') {
    return 'Filename fallback (speech-to-text unavailable)';
  }
  return source;
}
