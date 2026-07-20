import 'package:flutter/material.dart';

import '../models/cluster_run.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminClusterRunsScreen extends StatefulWidget {
  const AdminClusterRunsScreen({super.key});

  @override
  State<AdminClusterRunsScreen> createState() => _AdminClusterRunsScreenState();
}

class _AdminClusterRunsScreenState extends State<AdminClusterRunsScreen> {
  final _repository = AdminRepository();
  List<ClusterRunSummary> _items = [];
  String _algorithm = 'all';
  String _runAlgorithm = 'kmeans';
  String _source = 'youtube';
  final TextEditingController _limitController =
      TextEditingController(text: '12');
  final TextEditingController _nClustersController =
      TextEditingController(text: '3');
  final TextEditingController _maxFeaturesController =
      TextEditingController(text: '40');
  final TextEditingController _maxIterationsController =
      TextEditingController(text: '25');
  final TextEditingController _seedController =
      TextEditingController(text: '42');
  final TextEditingController _minClusterSizeController =
      TextEditingController(text: '2');
  final TextEditingController _minSamplesController =
      TextEditingController(text: '1');
  String? _error;
  bool _loading = false;
  bool _running = false;
  int _offset = 0;
  final int _limit = 12;
  int _total = 0;
  DatasetClusterRunResult? _latestRun;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _limitController.dispose();
    _nClustersController.dispose();
    _maxFeaturesController.dispose();
    _maxIterationsController.dispose();
    _seedController.dispose();
    _minClusterSizeController.dispose();
    _minSamplesController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await _repository.listClusterRuns(
        limit: _limit,
        offset: _offset,
        algorithm: _algorithm,
      );
      if (!mounted) return;
      setState(() {
        _items = response.items;
        _total = response.total;
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

  Future<void> _runClustering() async {
    setState(() {
      _running = true;
      _error = null;
    });
    try {
      final payload = <String, dynamic>{
        'source': _source,
        'algorithm': _runAlgorithm,
        'limit': int.tryParse(_limitController.text.trim()) ?? 12,
        'offset': 0,
        'n_clusters': int.tryParse(_nClustersController.text.trim()) ?? 3,
        'max_features': int.tryParse(_maxFeaturesController.text.trim()) ?? 40,
        'max_iterations':
            int.tryParse(_maxIterationsController.text.trim()) ?? 25,
        'seed': int.tryParse(_seedController.text.trim()) ?? 42,
        'min_cluster_size':
            int.tryParse(_minClusterSizeController.text.trim()) ?? 2,
        'min_samples': int.tryParse(_minSamplesController.text.trim()),
        'save_result': true,
      };
      final response = await _repository.runClusteringFromDataset(payload);
      if (!mounted) return;
      setState(() => _latestRun = response);
      await _load();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Clustering finished with ${_latestRun?.algorithm ?? _runAlgorithm}',
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    } finally {
      if (mounted) {
        setState(() => _running = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      title: 'Cluster Runs',
      currentRoute: '/admin-clusters',
      isAdmin: true,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Run clustering from dataset',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 16),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _runAlgorithm,
                                decoration: const InputDecoration(
                                    labelText: 'Run algorithm'),
                                items: const [
                                  DropdownMenuItem(
                                      value: 'kmeans', child: Text('KMeans')),
                                  DropdownMenuItem(
                                      value: 'hdbscan', child: Text('HDBSCAN')),
                                ],
                                onChanged: (value) {
                                  if (value == null) return;
                                  setState(() => _runAlgorithm = value);
                                },
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: DropdownButtonFormField<String>(
                                initialValue: _source,
                                decoration:
                                    const InputDecoration(labelText: 'Source'),
                                items: const [
                                  DropdownMenuItem(
                                      value: 'youtube', child: Text('YouTube')),
                                  DropdownMenuItem(
                                      value: 'google', child: Text('Google')),
                                ],
                                onChanged: (value) {
                                  if (value == null) return;
                                  setState(() => _source = value);
                                },
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: _limitController,
                                keyboardType: TextInputType.number,
                                decoration: const InputDecoration(
                                    labelText: 'Dataset limit'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: TextField(
                                controller: _maxFeaturesController,
                                keyboardType: TextInputType.number,
                                decoration: const InputDecoration(
                                    labelText: 'Max features'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        if (_runAlgorithm == 'kmeans')
                          Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _nClustersController,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                      labelText: 'Clusters'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: TextField(
                                  controller: _maxIterationsController,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                      labelText: 'Max iterations'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: TextField(
                                  controller: _seedController,
                                  keyboardType: TextInputType.number,
                                  decoration:
                                      const InputDecoration(labelText: 'Seed'),
                                ),
                              ),
                            ],
                          )
                        else
                          Row(
                            children: [
                              Expanded(
                                child: TextField(
                                  controller: _minClusterSizeController,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                      labelText: 'Min cluster size'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: TextField(
                                  controller: _minSamplesController,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                      labelText: 'Min samples'),
                                ),
                              ),
                            ],
                          ),
                        const SizedBox(height: 12),
                        FilledButton.icon(
                          onPressed: _running ? null : _runClustering,
                          icon: const Icon(Icons.play_arrow),
                          label:
                              Text(_running ? 'Running...' : 'Run clustering'),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                if (_latestRun != null)
                  Card(
                    child: ListTile(
                      leading: const Icon(Icons.analytics_outlined),
                      title: Text(
                        'Latest run: ${_latestRun?.algorithm ?? '-'}',
                      ),
                      subtitle: Text(
                        'run_id ${_latestRun?.runId ?? '-'} | items ${_latestRun?.totalItemsUsed ?? '-'} | '
                        'clusters ${_latestRun?.nClusters ?? '-'}',
                      ),
                    ),
                  ),
                Row(
                  children: [
                    Expanded(
                      child: DropdownButtonFormField<String>(
                        initialValue: _algorithm,
                        decoration:
                            const InputDecoration(labelText: 'Filter run list'),
                        items: const [
                          DropdownMenuItem(value: 'all', child: Text('All')),
                          DropdownMenuItem(
                              value: 'kmeans', child: Text('KMeans')),
                          DropdownMenuItem(
                              value: 'hdbscan', child: Text('HDBSCAN')),
                        ],
                        onChanged: (value) {
                          if (value == null) return;
                          setState(() {
                            _algorithm = value;
                            _offset = 0;
                          });
                          _load();
                        },
                      ),
                    ),
                    const SizedBox(width: 12),
                    IconButton(
                        onPressed: _load, icon: const Icon(Icons.refresh)),
                  ],
                ),
              ],
            ),
          ),
          if (_loading || _running) const LinearProgressIndicator(),
          Expanded(
            child: _error != null
                ? ErrorStateView(message: _error!, onRetry: _load)
                : _items.isEmpty
                    ? const EmptyStateView(
                        title: 'No cluster runs found',
                        message:
                            'Run clustering first to populate this section.',
                        icon: Icons.bubble_chart_outlined,
                      )
                    : RefreshIndicator(
                        onRefresh: _load,
                        child: ListView.builder(
                          padding: const EdgeInsets.all(16),
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
                              );
                            }
                            final item = _items[index];
                            return Card(
                              child: ListTile(
                                title: Text(
                                    'Run #${item.runId} | ${item.algorithm}'),
                                subtitle: Text(
                                  'clusters ${item.nClusters} | features ${item.featureDimension}\n'
                                  'members ${item.membershipCount} | inertia ${item.inertia}',
                                ),
                                isThreeLine: true,
                                trailing: const Icon(Icons.chevron_right),
                                onTap: () => Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) => AdminClusterRunDetailScreen(
                                        runId: item.runId),
                                  ),
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

class AdminClusterRunDetailScreen extends StatefulWidget {
  const AdminClusterRunDetailScreen({super.key, required this.runId});

  final int runId;

  @override
  State<AdminClusterRunDetailScreen> createState() =>
      _AdminClusterRunDetailScreenState();
}

class _AdminClusterRunDetailScreenState
    extends State<AdminClusterRunDetailScreen> {
  final _repository = AdminRepository();
  ClusterRunDetail? _data;
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
      final response = await _repository.getClusterRun(widget.runId);
      if (!mounted) return;
      setState(() => _data = response);
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
    final data = _data;
    final clusterBreakdown =
        data?.clusterBreakdown.map((item) => item.toChartItem()).toList() ??
            const [];
    final memberships = data?.recentMemberships ?? const [];
    return AppShell(
      title: 'Run #${widget.runId}',
      isAdmin: true,
      actions: [IconButton(onPressed: _load, icon: const Icon(Icons.refresh))],
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : _data == null
              ? const Center(child: CircularProgressIndicator())
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    if (_loading) const LinearProgressIndicator(),
                    Card(
                      child: ListTile(
                        title: Text(
                            '${data?.algorithm} | ${data?.nClusters} clusters'),
                        subtitle: Text(
                          'feature dimension ${data?.featureDimension} | '
                          'members ${data?.membershipCount} | inertia ${data?.inertia}',
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text('Cluster Breakdown',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 18)),
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: SimpleBarChart(
                          items: clusterBreakdown,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    const Text('Recent Memberships',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 18)),
                    ...memberships.map((row) {
                      return Card(
                        child: ListTile(
                          title: Text(row.clusterName),
                          subtitle: Text(
                            '${row.topTerms}\n${row.itemText}',
                            maxLines: 4,
                            overflow: TextOverflow.ellipsis,
                          ),
                          isThreeLine: true,
                        ),
                      );
                    }),
                  ],
                ),
    );
  }
}
