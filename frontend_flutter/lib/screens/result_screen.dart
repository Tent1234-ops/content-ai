import 'package:flutter/material.dart';

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

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_initialized) return;
    _initialized = true;
    final args =
        ModalRoute.of(context)?.settings.arguments as ResultScreenArgs?;
    if (args?.initialData != null) {
      _data = args!.initialData;
    } else if (args?.contentId != null) {
      _loadContent(args!.contentId!);
    }
  }

  Future<void> _loadContent(int contentId) async {
    try {
      final response = await _repository.getContentResult(contentId);
      if (!mounted) return;
      setState(() => _data = response);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    final data = _data;
    final recommendation = data?.recommendation;
    final classification = recommendation?.classification;
    final classifierConfidence = (classification?.confidence ?? 0) * 100;

    return AppShell(
      title: 'Analysis Result',
      isAdmin: AuthScope.of(context).isAdmin,
      child: _error != null
          ? ErrorStateView(message: _error!)
          : _data == null
              ? const Center(child: CircularProgressIndicator())
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    Text(data?.title ?? 'Result',
                        style: Theme.of(context).textTheme.titleLarge),
                    const SizedBox(height: 8),
                    if (data?.transcript.isNotEmpty ?? false)
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(16),
                          child: Text(
                            data?.transcript ?? '',
                            maxLines: 4,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ),
                    const SizedBox(height: 8),
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(Icons.account_tree_outlined),
                                const SizedBox(width: 8),
                                Text(
                                  'Dataset Classifier',
                                  style:
                                      Theme.of(context).textTheme.titleMedium,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Predicted domain: ${classification?.domain ?? recommendation?.domain ?? data?.fallbackDomain ?? '-'}',
                              style:
                                  const TextStyle(fontWeight: FontWeight.bold),
                            ),
                            const SizedBox(height: 8),
                            Wrap(
                              spacing: 8,
                              runSpacing: 8,
                              children: [
                                Chip(
                                  avatar: const Icon(Icons.insights, size: 18),
                                  label: Text(
                                      'confidence ${classifierConfidence.toStringAsFixed(1)}%'),
                                ),
                                Chip(
                                  avatar: const Icon(Icons.rule, size: 18),
                                  label: Text(
                                      'rule ${classification?.ruleDomain ?? '-'}'),
                                ),
                                Chip(
                                  avatar: const Icon(Icons.storage_outlined,
                                      size: 18),
                                  label: Text(
                                      '${classification?.source ?? 'youtube'} profile'),
                                ),
                              ],
                            ),
                            const SizedBox(height: 12),
                            if (classification?.candidates.isNotEmpty ?? false)
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text('Top classifier candidates'),
                                  const SizedBox(height: 8),
                                  ...classification!.candidates
                                      .take(3)
                                      .map((row) {
                                    final matchedTerms =
                                        row.matchedTerms.isEmpty
                                            ? '-'
                                            : row.matchedTerms.join(', ');
                                    return ListTile(
                                      dense: true,
                                      contentPadding: EdgeInsets.zero,
                                      title: Text(
                                          '${row.domain} | score ${row.score}'),
                                      subtitle: Text(
                                          'samples ${row.sampleSize} | matched $matchedTerms'),
                                    );
                                  }),
                                ],
                              ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(Icons.auto_awesome_outlined),
                                const SizedBox(width: 8),
                                Text(
                                  'Keyword Gap Recommendation',
                                  style:
                                      Theme.of(context).textTheme.titleMedium,
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                                'Domain used for gap: ${recommendation?.domain ?? data?.fallbackDomain ?? '-'}'),
                            const SizedBox(height: 8),
                            Text(
                                'Recommended duration: ${recommendation?.duration.recommendedRange ?? '-'}'),
                            const SizedBox(height: 8),
                            Text(
                                'Profile sample size: ${recommendation?.duration.sampleSize ?? 0}'),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text('Missing Keywords',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    ...(recommendation?.missingKeywords ?? const []).map((row) {
                      return ListTile(
                        dense: true,
                        title: Text(row.keyword),
                        trailing: Text(row.score.toString()),
                      );
                    }),
                    const SizedBox(height: 12),
                    const Text('Hook Suggestions',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    ...(recommendation?.hookKeywords ?? const []).map((row) {
                      return ListTile(
                        dense: true,
                        title: Text(row.keyword),
                        trailing: Text(row.score.toString()),
                      );
                    }),
                    const SizedBox(height: 12),
                    const Text('Missing Dimensions',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    ...(recommendation?.missingDimensions ?? const [])
                        .map((row) {
                      return ListTile(
                        dense: true,
                        title: Text(row.name),
                        subtitle: Text('Status: ${row.userStatus}'),
                        trailing: Text(row.score.toString()),
                      );
                    }),
                  ],
                ),
    );
  }
}
