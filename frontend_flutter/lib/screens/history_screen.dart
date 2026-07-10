import 'package:flutter/material.dart';

import '../models/content_history.dart';
import '../repositories/content_repository.dart';
import '../state/auth_scope.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';
import 'result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _repository = ContentRepository();
  List<ContentHistoryItem> _items = [];
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
      final response = await _repository.listMyContents(limit: 20);
      if (!mounted) return;
      setState(() => _items = response.items);
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
    final auth = AuthScope.of(context);
    return AppShell(
      title: 'History / Ideas',
      currentRoute: '/history',
      isAdmin: auth.isAdmin,
      child: _error != null
          ? ErrorStateView(message: _error!, onRetry: _load)
          : _items.isEmpty && !_loading
              ? const EmptyStateView(
                  title: 'No analysis history yet',
                  message:
                      'Analyze your first clip and the saved recommendations will appear here.',
                  icon: Icons.history_outlined,
                )
              : RefreshIndicator(
                  onRefresh: _load,
                  child: ListView.builder(
                    itemCount: _items.length + (_loading ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (_loading && index == 0) {
                        return const LinearProgressIndicator();
                      }
                      final itemIndex = _loading ? index - 1 : index;
                      final item = _items[itemIndex];
                      final recommendedKeywords =
                          item.recommendedKeywords.join(', ');
                      return Card(
                        child: ListTile(
                          title: Text(item.title),
                          subtitle: Text(
                            '${item.domain}\n'
                            '${item.transcriptPreview}\n'
                            'keywords: ${recommendedKeywords.isEmpty ? '-' : recommendedKeywords}',
                          ),
                          isThreeLine: true,
                          trailing: Text(item.recommendedDuration),
                          onTap: () => Navigator.pushNamed(
                            context,
                            '/result',
                            arguments:
                                ResultScreenArgs(contentId: item.contentId),
                          ),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}
