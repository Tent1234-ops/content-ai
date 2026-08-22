import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../models/dataset_review.dart';
import '../repositories/admin_repository.dart';
import '../utils/notebooklm_markdown_parser.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminTranscriptImportScreen extends StatefulWidget {
  const AdminTranscriptImportScreen({super.key, this.repository});

  final AdminRepository? repository;

  @override
  State<AdminTranscriptImportScreen> createState() =>
      _AdminTranscriptImportScreenState();
}

class _AdminTranscriptImportScreenState
    extends State<AdminTranscriptImportScreen> {
  final _formKey = GlobalKey<FormState>();
  final _videoUrlController = TextEditingController();
  final _transcriptController = TextEditingController();
  late final AdminRepository _repository;

  List<DatasetReviewTaxonomyLeaf> _taxonomy = const [];
  String? _leafKey;
  String _language = 'th';
  String _captionType = 'unspecified';
  String _strategy = 'classification_diverse';
  int? _batchRunId;
  NotebookLMImportResult? _lastResult;
  String? _error;
  String? _transcriptFileName;
  int? _transcriptFileCharacters;
  bool _loadingTaxonomy = true;
  bool _submitting = false;
  bool _readingMarkdown = false;

  @override
  void initState() {
    super.initState();
    _repository = widget.repository ?? AdminRepository();
    _loadTaxonomy();
  }

  @override
  void dispose() {
    _videoUrlController.dispose();
    _transcriptController.dispose();
    super.dispose();
  }

  Future<void> _loadTaxonomy() async {
    try {
      final queue = await _repository.listDatasetReviewQueue(
        limit: 1,
        offset: 0,
        status: 'all',
      );
      if (!mounted) return;
      setState(() {
        _taxonomy = queue.taxonomy;
        _leafKey = _leafKey ??
            (queue.taxonomy.isEmpty ? null : queue.taxonomy.first.leafKey);
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _loadingTaxonomy = false);
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate() || _leafKey == null) return;
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      final result = await _repository.createNotebookLMCandidate(
        videoUrl: _videoUrlController.text,
        transcript: _transcriptController.text,
        proposedLeafKey: _leafKey!,
        transcriptLanguage: _language,
        captionType: _captionType,
        collectionStrategy: _strategy,
        collectionRunId: _batchRunId,
      );
      if (!mounted) return;
      setState(() {
        _batchRunId = result.collectionRunId;
        _lastResult = result;
        _videoUrlController.clear();
        _transcriptController.clear();
        _transcriptFileName = null;
        _transcriptFileCharacters = null;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Candidate validated and added to review')),
      );
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _pickMarkdownTranscript() async {
    setState(() {
      _readingMarkdown = true;
      _error = null;
    });
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['md'],
        allowMultiple: false,
        withData: true,
      );
      if (result == null) return;

      final file = result.files.single;
      final bytes = file.bytes;
      if (bytes == null) {
        throw const FormatException('The selected file could not be read.');
      }
      if (bytes.length > 4 * 1024 * 1024) {
        throw const FormatException(
            'The Markdown file must be 4 MB or smaller.');
      }

      final markdown = utf8.decode(bytes, allowMalformed: false);
      final document = NotebookLmMarkdownParser.parse(markdown);
      if (!mounted) return;
      setState(() {
        _transcriptController.text = document.transcript;
        _transcriptFileName = file.name;
        _transcriptFileCharacters = document.transcript.length;
        if (_videoUrlController.text.trim().isEmpty &&
            document.sourceUrl != null) {
          _videoUrlController.text = document.sourceUrl!;
        }
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Loaded ${document.transcript.length} transcript characters',
          ),
        ),
      );
    } on FormatException catch (error) {
      if (mounted) setState(() => _error = error.message.toString());
    } catch (error) {
      if (mounted) {
        setState(() => _error = 'Could not import Markdown: $error');
      }
    } finally {
      if (mounted) setState(() => _readingMarkdown = false);
    }
  }

  void _clearMarkdownTranscript() {
    setState(() {
      _transcriptController.clear();
      _transcriptFileName = null;
      _transcriptFileCharacters = null;
    });
  }

  void _startNewBatch() {
    setState(() {
      _batchRunId = null;
      _lastResult = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      title: 'Transcript Import',
      currentRoute: '/admin-transcript-import',
      isAdmin: true,
      actions: [
        IconButton(
          onPressed: _submitting ? null : _startNewBatch,
          icon: const Icon(Icons.create_new_folder_outlined),
          tooltip: 'Start new batch',
        ),
        IconButton(
          onPressed: () => Navigator.pushNamed(
            context,
            '/admin-dataset-review',
          ),
          icon: const Icon(Icons.fact_check_outlined),
          tooltip: 'Open dataset review',
        ),
      ],
      child: _loadingTaxonomy
          ? const Center(child: CircularProgressIndicator())
          : _taxonomy.isEmpty
              ? ErrorStateView(
                  message: 'No active taxonomy categories were found.',
                  onRetry: _loadTaxonomy,
                )
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 1000),
                        child: Form(
                          key: _formKey,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              _BatchBand(
                                runId: _batchRunId,
                                candidateCount:
                                    _lastResult?.candidateCount ?? 0,
                              ),
                              if (_error != null) ...[
                                const SizedBox(height: 12),
                                MaterialBanner(
                                  content: Text(_error!),
                                  actions: [
                                    TextButton(
                                      onPressed: () =>
                                          setState(() => _error = null),
                                      child: const Text('Dismiss'),
                                    ),
                                  ],
                                ),
                              ],
                              const SizedBox(height: 16),
                              TextFormField(
                                controller: _videoUrlController,
                                decoration: const InputDecoration(
                                  labelText: 'YouTube Creative Commons URL',
                                  prefixIcon: Icon(Icons.link),
                                ),
                                keyboardType: TextInputType.url,
                                validator: (value) {
                                  final raw = value?.trim() ?? '';
                                  if (raw.isEmpty) {
                                    return 'YouTube URL is required';
                                  }
                                  final uri = Uri.tryParse(raw);
                                  if (uri == null || !uri.hasScheme) {
                                    return 'Enter a valid YouTube URL';
                                  }
                                  return null;
                                },
                              ),
                              const SizedBox(height: 12),
                              LayoutBuilder(
                                builder: (context, constraints) {
                                  final width = constraints.maxWidth < 720
                                      ? constraints.maxWidth
                                      : (constraints.maxWidth - 24) / 3;
                                  return Wrap(
                                    spacing: 12,
                                    runSpacing: 12,
                                    children: [
                                      SizedBox(
                                        width: width,
                                        child: DropdownButtonFormField<String>(
                                          initialValue: _leafKey,
                                          isExpanded: true,
                                          decoration: const InputDecoration(
                                            labelText: 'Proposed category',
                                          ),
                                          items: _taxonomy
                                              .map(
                                                (leaf) => DropdownMenuItem(
                                                  value: leaf.leafKey,
                                                  child: Text(
                                                    leaf.path,
                                                    overflow:
                                                        TextOverflow.ellipsis,
                                                  ),
                                                ),
                                              )
                                              .toList(),
                                          onChanged: (value) =>
                                              setState(() => _leafKey = value),
                                        ),
                                      ),
                                      SizedBox(
                                        width: width,
                                        child: DropdownButtonFormField<String>(
                                          initialValue: _language,
                                          decoration: const InputDecoration(
                                            labelText: 'Transcript language',
                                          ),
                                          items: const [
                                            DropdownMenuItem(
                                              value: 'th',
                                              child: Text('Thai'),
                                            ),
                                            DropdownMenuItem(
                                              value: 'en',
                                              child: Text('English'),
                                            ),
                                          ],
                                          onChanged: (value) {
                                            if (value != null) {
                                              setState(() => _language = value);
                                            }
                                          },
                                        ),
                                      ),
                                      SizedBox(
                                        width: width,
                                        child: DropdownButtonFormField<String>(
                                          initialValue: _captionType,
                                          isExpanded: true,
                                          decoration: const InputDecoration(
                                            labelText: 'Caption type',
                                          ),
                                          items: const [
                                            DropdownMenuItem(
                                              value: 'unspecified',
                                              child: Text('Not disclosed'),
                                            ),
                                            DropdownMenuItem(
                                              value: 'manual',
                                              child: Text('Manual captions'),
                                            ),
                                            DropdownMenuItem(
                                              value: 'auto_generated',
                                              child: Text('Auto-generated'),
                                            ),
                                          ],
                                          onChanged: (value) {
                                            if (value != null) {
                                              setState(
                                                () => _captionType = value,
                                              );
                                            }
                                          },
                                        ),
                                      ),
                                    ],
                                  );
                                },
                              ),
                              const SizedBox(height: 12),
                              SegmentedButton<String>(
                                segments: const [
                                  ButtonSegment(
                                    value: 'classification_diverse',
                                    icon: Icon(Icons.category_outlined),
                                    label: Text('Classification'),
                                  ),
                                  ButtonSegment(
                                    value: 'recommendation_high_performance',
                                    icon: Icon(Icons.trending_up),
                                    label: Text('High performance'),
                                  ),
                                ],
                                selected: {_strategy},
                                onSelectionChanged: (values) =>
                                    setState(() => _strategy = values.first),
                              ),
                              const SizedBox(height: 12),
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  OutlinedButton.icon(
                                    onPressed: _readingMarkdown || _submitting
                                        ? null
                                        : _pickMarkdownTranscript,
                                    icon: _readingMarkdown
                                        ? const SizedBox.square(
                                            dimension: 18,
                                            child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                            ),
                                          )
                                        : const Icon(
                                            Icons.upload_file_outlined),
                                    label: Text(
                                      _readingMarkdown
                                          ? 'Reading file'
                                          : 'Import .md',
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: _transcriptFileName == null
                                        ? const SizedBox.shrink()
                                        : ListTile(
                                            dense: true,
                                            contentPadding: EdgeInsets.zero,
                                            leading: const Icon(
                                              Icons.description_outlined,
                                            ),
                                            title: Text(
                                              _transcriptFileName!,
                                              maxLines: 2,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                            subtitle: Text(
                                              '$_transcriptFileCharacters transcript characters',
                                            ),
                                            trailing: IconButton(
                                              onPressed:
                                                  _clearMarkdownTranscript,
                                              icon: const Icon(Icons.close),
                                              tooltip: 'Remove imported file',
                                            ),
                                          ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              TextFormField(
                                controller: _transcriptController,
                                minLines: 12,
                                maxLines: 24,
                                decoration: const InputDecoration(
                                  labelText: 'Full source transcript',
                                  alignLabelWithHint: true,
                                ),
                                validator: (value) {
                                  final raw = value?.trim() ?? '';
                                  if (raw.length < 80) {
                                    return 'Paste the complete transcript (at least 80 characters)';
                                  }
                                  return null;
                                },
                              ),
                              const SizedBox(height: 16),
                              FilledButton.icon(
                                onPressed: _submitting ? null : _submit,
                                icon: _submitting
                                    ? const SizedBox.square(
                                        dimension: 18,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      )
                                    : const Icon(Icons.playlist_add_check),
                                label: Text(
                                  _submitting
                                      ? 'Validating'
                                      : 'Validate and add candidate',
                                ),
                              ),
                              if (_lastResult != null) ...[
                                const SizedBox(height: 16),
                                _ValidatedCandidateBand(
                                  result: _lastResult!,
                                  onReview: () => Navigator.pushNamed(
                                    context,
                                    '/admin-dataset-review',
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
    );
  }
}

class _BatchBand extends StatelessWidget {
  const _BatchBand({required this.runId, required this.candidateCount});

  final int? runId;
  final int candidateCount;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.inventory_2_outlined),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              runId == null
                  ? 'New NotebookLM import batch'
                  : 'Batch run $runId',
              style: Theme.of(context).textTheme.titleSmall,
            ),
          ),
          if (runId != null) Text('$candidateCount candidates'),
        ],
      ),
    );
  }
}

class _ValidatedCandidateBand extends StatelessWidget {
  const _ValidatedCandidateBand({required this.result, required this.onReview});

  final NotebookLMImportResult result;
  final VoidCallback onReview;

  String _duration(int seconds) {
    final hours = seconds ~/ 3600;
    final minutes = (seconds % 3600) ~/ 60;
    final remaining = seconds % 60;
    if (hours > 0) return '${hours}h ${minutes}m ${remaining}s';
    return '${minutes}m ${remaining}s';
  }

  @override
  Widget build(BuildContext context) {
    final candidate = result.candidate;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.green.shade700),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.verified_outlined, color: Colors.green.shade700),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  candidate.title,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '${candidate.channelTitle} | '
            '${_duration(candidate.durationSeconds)}',
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              Chip(label: Text(candidate.proposedLeafKey)),
              Chip(label: Text(candidate.transcriptLanguage.toUpperCase())),
              const Chip(label: Text('Full video transcript')),
              const Chip(label: Text('Creative Commons verified')),
            ],
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: onReview,
            icon: const Icon(Icons.fact_check_outlined),
            label: const Text('Open review queue'),
          ),
        ],
      ),
    );
  }
}
