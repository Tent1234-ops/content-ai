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
        if (item.category.isNotEmpty) {
          discoveredCategories.add(item.category);
        }
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
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  void _applyFilters() {
    setState(() => _offset = 0);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      title: 'Admin Datasets',
      currentRoute: '/admin-datasets',
      isAdmin: true,
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
                    IconButton(
                        onPressed: _load, icon: const Icon(Icons.refresh)),
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
                          DropdownMenuItem(
                              value: 'youtube', child: Text('YouTube')),
                          DropdownMenuItem(
                              value: 'google', child: Text('Google')),
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
                        decoration:
                            const InputDecoration(labelText: 'Category'),
                        items: _categories
                            .map((value) => DropdownMenuItem(
                                value: value, child: Text(value)))
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
                        message:
                            'Try changing the source, category, or search term.',
                      )
                    : RefreshIndicator(
                        onRefresh: _load,
                        child: ListView.builder(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          itemCount: _items.length + 1,
                          itemBuilder: (context, index) {
                            if (index == _items.length) {
                              return Padding(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 8),
                                child: PaginationBar(
                                  offset: _offset,
                                  limit: _limit,
                                  total: _total,
                                  onPrevious: _offset <= 0
                                      ? null
                                      : () {
                                          setState(() => _offset =
                                              (_offset - _limit)
                                                  .clamp(0, _offset));
                                          _load();
                                        },
                                  onNext: _offset + _limit >= _total
                                      ? null
                                      : () {
                                          setState(() => _offset += _limit);
                                          _load();
                                        },
                                ),
                              );
                            }
                            final item = _items[index];
                            return Card(
                              child: ListTile(
                                title: Text(item.title),
                                subtitle: Text(
                                  '${item.sourcePlatform} | ${item.category}\n'
                                  'views ${item.views} likes ${item.likes} comments ${item.comments}',
                                ),
                                isThreeLine: true,
                                trailing: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const Text('score'),
                                    Text('${item.trendScore}'),
                                  ],
                                ),
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
