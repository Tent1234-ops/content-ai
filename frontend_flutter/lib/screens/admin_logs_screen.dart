import 'package:flutter/material.dart';

import '../models/system_log.dart';
import '../repositories/admin_repository.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';

class AdminLogsScreen extends StatefulWidget {
  const AdminLogsScreen({super.key});

  @override
  State<AdminLogsScreen> createState() => _AdminLogsScreenState();
}

class _AdminLogsScreenState extends State<AdminLogsScreen> {
  final _repository = AdminRepository();
  final _actionController = TextEditingController();
  List<SystemLogItem> _items = [];
  String _status = 'all';
  String? _error;
  bool _loading = false;
  int _offset = 0;
  final int _limit = 16;
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
      final response = await _repository.listLogs(
        limit: _limit,
        offset: _offset,
        status: _status,
        action: _actionController.text,
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

  Color _statusColor(String status, BuildContext context) {
    switch (status.toLowerCase()) {
      case 'success':
        return Colors.green;
      case 'failed':
      case 'error':
        return Theme.of(context).colorScheme.error;
      default:
        return Theme.of(context).colorScheme.primary;
    }
  }

  void _applyFilters() {
    setState(() => _offset = 0);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      title: 'System Logs',
      currentRoute: '/admin-logs',
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
                        controller: _actionController,
                        decoration: const InputDecoration(
                          labelText: 'Search action',
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
                DropdownButtonFormField<String>(
                  initialValue: _status,
                  decoration: const InputDecoration(labelText: 'Status'),
                  items: const [
                    DropdownMenuItem(value: 'all', child: Text('All')),
                    DropdownMenuItem(value: 'success', child: Text('Success')),
                    DropdownMenuItem(value: 'failed', child: Text('Failed')),
                    DropdownMenuItem(value: 'error', child: Text('Error')),
                  ],
                  onChanged: (value) {
                    if (value == null) return;
                    setState(() => _status = value);
                    _applyFilters();
                  },
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
                        title: 'No logs found',
                        message: 'Try another status or action filter.',
                        icon: Icons.receipt_long_outlined,
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
                            final status = item.status;
                            return Card(
                              child: ListTile(
                                title: Text(item.action),
                                subtitle: Text(item.detail),
                                trailing: Chip(
                                  label: Text(status),
                                  backgroundColor: _statusColor(status, context)
                                      .withValues(alpha: 0.12),
                                  side: BorderSide(
                                      color: _statusColor(status, context)),
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
