import 'package:flutter/material.dart';

import '../models/dataset_item.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminDatasetsScreen extends StatefulWidget {
  const AdminDatasetsScreen({super.key});

  @override
  State<AdminDatasetsScreen> createState() => _AdminDatasetsScreenState();
}

class _AdminDatasetsScreenState extends State<AdminDatasetsScreen> {
  final _repository = AdminRepository();
  final _searchController = TextEditingController();
  List<DatasetItem> _items = [];
  List<String> _categories = ['all'];
  String _source = 'all';
  String _category = 'all';
  String? _error;
  bool _loading = false;
  int _offset = 0;
  final int _limit = 12;
  int _total = 0;

  @override
  void initState() {
    super.initState();
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
      final response = await _repository.listDatasets(
        limit: _limit,
        offset: _offset,
        source: _source,
        category: _category,
        search: _searchController.text,
      );
      if (!mounted) return;
      final discoveredCategories = <String>{'all'};
      for (final item in response.items) {
        if (item.category.isNotEmpty) discoveredCategories.add(item.category);
      }
      setState(() {
        _items = response.items;
        _total = response.total;
        _categories = discoveredCategories.toList()..sort();
      });
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

  Future<void> _openEditor([DatasetItem? item]) async {
    final saved = await showDialog<bool>(
      context: context,
      builder: (context) => _DatasetEditorDialog(item: item),
    );
    if (saved == true) {
      await _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      title: 'Admin Datasets',
      currentRoute: '/admin-datasets',
      isAdmin: true,
      actions: [
        IconButton(
          onPressed: () => _openEditor(),
          icon: const Icon(Icons.add),
          tooltip: 'Add dataset',
        ),
        IconButton(onPressed: _load, icon: const Icon(Icons.refresh)),
      ],
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _searchController,
                        decoration: const InputDecoration(
                          labelText: 'Search title or transcript',
                          prefixIcon: Icon(Icons.search),
                        ),
                        onSubmitted: (_) => _applyFilters(),
                      ),
                    ),
                    const SizedBox(width: 12),
                    FilledButton.icon(
                      onPressed: () => _openEditor(),
                      icon: const Icon(Icons.add),
                      label: const Text('Add'),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        initialValue: _source,
                        decoration: const InputDecoration(labelText: 'Source'),
                        items: const [
                          DropdownMenuItem(value: 'all', child: Text('All')),
                          DropdownMenuItem(value: 'youtube', child: Text('YouTube')),
                          DropdownMenuItem(value: 'google', child: Text('Google')),
                          DropdownMenuItem(value: 'tiktok', child: Text('TikTok')),
                        ],
                        onChanged: (value) {
                          if (value == null) return;
                          setState(() => _source = value);
                          _applyFilters();
                        },
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        initialValue: _category,
                        decoration: const InputDecoration(labelText: 'Category'),
                        items: _categories
                            .map((value) => DropdownMenuItem(value: value, child: Text(value)))
                            .toList(),
                        onChanged: (value) {
                          if (value == null) return;
                          setState(() => _category = value);
                          _applyFilters();
                        },
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          if (_loading) const LinearProgressIndicator(),
          Expanded(
            child: _error != null
                ? ErrorStateView(message: _error!, onRetry: _load)
                : _items.isEmpty
                    ? const EmptyStateView(
                        title: 'No datasets found',
                        message: 'Add a dataset or adjust filters.',
                        icon: Icons.storage_outlined,
                      )
                    : RefreshIndicator(
                        onRefresh: _load,
                        child: ListView.builder(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          itemCount: _items.length + 1,
                          itemBuilder: (context, index) {
                            if (index == _items.length) {
                              return PaginationBar(
                                offset: _offset,
                                limit: _limit,
                                total: _total,
                                onPrevious: _offset <= 0
                                    ? null
                                    : () {
                                        setState(() => _offset = (_offset - _limit).clamp(0, _offset));
                                        _load();
                                      },
                                onNext: _offset + _limit >= _total
                                    ? null
                                    : () {
                                        setState(() => _offset += _limit);
                                        _load();
                                      },
                              );
                            }
                            final item = _items[index];
                            return Card(
                              child: ListTile(
                                leading: const Icon(Icons.dataset_outlined),
                                title: Text(item.title),
                                subtitle: Text(
                                  '${item.sourcePlatform} | ${item.category}\n'
                                  'views ${item.views} likes ${item.likes} comments ${item.comments} | '
                                  'duration ${item.durationSeconds ?? 0}s',
                                ),
                                isThreeLine: true,
                                trailing: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const Text('score'),
                                    Text(item.trendScore.toStringAsFixed(1)),
                                  ],
                                ),
                                onTap: () => _openEditor(item),
                              ),
                            );
                          },
                        ),
                      ),
          ),
        ],
      ),
    );
  }
}

class _DatasetEditorDialog extends StatefulWidget {
  const _DatasetEditorDialog({this.item});

  final DatasetItem? item;

  @override
  State<_DatasetEditorDialog> createState() => _DatasetEditorDialogState();
}

class _DatasetEditorDialogState extends State<_DatasetEditorDialog> {
  final _repository = AdminRepository();
  late final TextEditingController _title;
  late final TextEditingController _url;
  late final TextEditingController _transcript;
  late final TextEditingController _category;
  late final TextEditingController _source;
  late final TextEditingController _views;
  late final TextEditingController _likes;
  late final TextEditingController _comments;
  late final TextEditingController _score;
  late final TextEditingController _duration;
  bool _saving = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    final item = widget.item;
    _title = TextEditingController(text: item?.title ?? '');
    _url = TextEditingController(text: item?.videoUrl ?? '');
    _transcript = TextEditingController(text: item?.transcript ?? '');
    _category = TextEditingController(text: item?.category ?? '');
    _source = TextEditingController(text: item?.sourcePlatform ?? 'youtube_admin');
    _views = TextEditingController(text: '${item?.views ?? 0}');
    _likes = TextEditingController(text: '${item?.likes ?? 0}');
    _comments = TextEditingController(text: '${item?.comments ?? 0}');
    _score = TextEditingController(text: '${item?.trendScore ?? 0}');
    _duration = TextEditingController(text: '${item?.durationSeconds ?? ''}');
  }

  @override
  void dispose() {
    _title.dispose();
    _url.dispose();
    _transcript.dispose();
    _category.dispose();
    _source.dispose();
    _views.dispose();
    _likes.dispose();
    _comments.dispose();
    _score.dispose();
    _duration.dispose();
    super.dispose();
  }

  Map<String, dynamic> _payload() {
    final durationText = _duration.text.trim();
    return {
      'title': _title.text.trim(),
      'video_url': _url.text.trim().isEmpty ? null : _url.text.trim(),
      'transcript': _transcript.text.trim().isEmpty ? null : _transcript.text.trim(),
      'category': _category.text.trim().isEmpty ? null : _category.text.trim(),
      'source_platform': _source.text.trim().isEmpty ? 'youtube_admin' : _source.text.trim(),
      'views': int.tryParse(_views.text.trim()) ?? 0,
      'likes': int.tryParse(_likes.text.trim()) ?? 0,
      'comments': int.tryParse(_comments.text.trim()) ?? 0,
      'trend_score': double.tryParse(_score.text.trim()) ?? 0,
      'duration_seconds': durationText.isEmpty ? null : int.tryParse(durationText),
    };
  }

  Future<void> _save() async {
    if (_title.text.trim().isEmpty) {
      setState(() => _error = 'Title is required');
      return;
    }
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      if (widget.item == null) {
        await _repository.createDataset(_payload());
      } else {
        await _repository.updateDataset(widget.item!.datasetId, _payload());
      }
      if (!mounted) return;
      Navigator.pop(context, true);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isEdit = widget.item != null;
    return AlertDialog(
      title: Text(isEdit ? 'Edit dataset' : 'Add dataset'),
      content: SizedBox(
        width: 640,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                ),
              TextField(controller: _title, decoration: const InputDecoration(labelText: 'Title')),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(child: TextField(controller: _source, decoration: const InputDecoration(labelText: 'Source platform'))),
                  const SizedBox(width: 10),
                  Expanded(child: TextField(controller: _category, decoration: const InputDecoration(labelText: 'Category'))),
                ],
              ),
              const SizedBox(height: 10),
              TextField(controller: _url, decoration: const InputDecoration(labelText: 'Video URL')),
              const SizedBox(height: 10),
              TextField(
                controller: _transcript,
                decoration: const InputDecoration(labelText: 'Transcript'),
                minLines: 3,
                maxLines: 6,
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(child: TextField(controller: _views, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Views'))),
                  const SizedBox(width: 10),
                  Expanded(child: TextField(controller: _likes, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Likes'))),
                  const SizedBox(width: 10),
                  Expanded(child: TextField(controller: _comments, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Comments'))),
                ],
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(child: TextField(controller: _score, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Trend score'))),
                  const SizedBox(width: 10),
                  Expanded(child: TextField(controller: _duration, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Duration seconds'))),
                ],
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(onPressed: _saving ? null : () => Navigator.pop(context, false), child: const Text('Cancel')),
        FilledButton.icon(
          onPressed: _saving ? null : _save,
          icon: _saving
              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.save_outlined),
          label: Text(_saving ? 'Saving...' : 'Save'),
        ),
      ],
    );
  }
}
