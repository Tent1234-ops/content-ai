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
  bool _saveLoading = false;

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

  Future<void> _saveToIdeas() async {
    setState(() => _saveLoading = true);
    try {
      // TODO: Implement save to ideas
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Saved to My Ideas!'),
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
    final domain = classification?.domain ?? recommendation?.domain ?? data?.fallbackDomain ?? '-';
    final missingKeywords = recommendation?.missingKeywords ?? const [];
    final hookKeywords = recommendation?.hookKeywords ?? const [];
    final duration = recommendation?.duration;

    return AppShell(
      title: 'Analysis Result',
      isAdmin: AuthScope.of(context).isAdmin,
      actions: [
        if (!_saveLoading)
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
                   // Refresh functionality
                 },
                 child: ListView(
                   padding: const EdgeInsets.all(16),
                   children: [
                     // Header
                     Column(
                       crossAxisAlignment: CrossAxisAlignment.start,
                       children: [
                         Text(
                           data?.title ?? 'Result',
                           style: Theme.of(context).textTheme.headlineSmall,
                           maxLines: 2,
                           overflow: TextOverflow.ellipsis,
                         ),
                         const SizedBox(height: 8),
                         Row(
                           children: [
                             Chip(
                               avatar: const Icon(Icons.domain, size: 18),
                               label: Text(domain),
                             ),
                             const SizedBox(width: 8),
                             Chip(
                               avatar: const Icon(Icons.check_circle, size: 18),
                               label: Text('${classifierConfidence.toStringAsFixed(0)}% confidence'),
                             ),
                           ],
                         ),
                       ],
                     ),
                     const SizedBox(height: 24),

                     // Transcript Preview
                     if ((data?.transcript.isNotEmpty ?? false))
                       Column(
                         crossAxisAlignment: CrossAxisAlignment.start,
                         children: [
                           _SectionHeader(
                             title: 'Transcript Preview',
                             icon: Icons.subtitles_outlined,
                           ),
                           Card(
                             child: Padding(
                               padding: const EdgeInsets.all(16),
                               child: Text(
                                 data?.transcript ?? '',
                                 maxLines: 5,
                                 overflow: TextOverflow.ellipsis,
                                 style: Theme.of(context).textTheme.bodyMedium,
                               ),
                             ),
                           ),
                           const SizedBox(height: 24),
                         ],
                       ),

                     // Classification Detail
                     _SectionHeader(
                       title: 'Content Classification',
                       icon: Icons.account_tree_outlined,
                     ),
                     Card(
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
                                         'Predicted Domain',
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
                                       '${classifierConfidence.toStringAsFixed(0)}%',
                                       style: Theme.of(context)
                                           .textTheme
                                           .headlineSmall
                                           ?.copyWith(color: Theme.of(context).primaryColor),
                                     ),
                                     Text(
                                       'Confidence',
                                       style: Theme.of(context).textTheme.labelSmall,
                                     ),
                                   ],
                                 ),
                               ],
                             ),
                             const SizedBox(height: 16),
                             const Divider(height: 1),
                             const SizedBox(height: 12),
                             if (classification?.candidates.isNotEmpty ?? false)
                               Column(
                                 crossAxisAlignment: CrossAxisAlignment.start,
                                 children: [
                                   Text(
                                     'Alternative Domains',
                                     style: Theme.of(context).textTheme.labelMedium,
                                   ),
                                   const SizedBox(height: 8),
                                   ...classification!.candidates.take(3).map((candidate) {
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
                                               value: candidate.score,
                                               minHeight: 4,
                                             ),
                                           ),
                                         ],
                                       ),
                                     );
                                   }),
                                 ],
                               ),
                           ],
                         ),
                       ),
                     ),
                     const SizedBox(height: 24),

                     // Duration Recommendation
                     if (duration != null)
                       Column(
                         crossAxisAlignment: CrossAxisAlignment.start,
                         children: [
                           _SectionHeader(
                             title: 'Recommended Video Duration',
                             icon: Icons.schedule_outlined,
                           ),
                           Card(
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
                                             'Based on',
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
                           ),
                           const SizedBox(height: 24),
                         ],
                       ),

                     // Hook Keywords
                     if (hookKeywords.isNotEmpty)
                       Column(
                         crossAxisAlignment: CrossAxisAlignment.start,
                         children: [
                           _SectionHeader(
                             title: 'Hook Suggestions (First 60s)',
                             icon: Icons.lightbulb_outline,
                           ),
                           Card(
                             child: Padding(
                               padding: const EdgeInsets.all(16),
                               child: Column(
                                 children: hookKeywords.map((keyword) {
                                   return Padding(
                                     padding: const EdgeInsets.only(bottom: 12),
                                     child: Row(
                                       mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                       children: [
                                         Expanded(
                                           child: Column(
                                             crossAxisAlignment: CrossAxisAlignment.start,
                                             children: [
                                               Text(
                                                 keyword.keyword,
                                                 style: Theme.of(context).textTheme.labelMedium,
                                               ),
                                             ],
                                           ),
                                         ),
                                         Chip(
                                           label: Text('${keyword.score.toStringAsFixed(2)}'),
                                           side: const BorderSide(color: Color(0xFFE0E0E0)),
                                           backgroundColor: Colors.transparent,
                                         ),
                                       ],
                                     ),
                                   );
                                 }).toList(),
                               ),
                             ),
                           ),
                           const SizedBox(height: 24),
                         ],
                       ),

                     // Missing Keywords
                     if (missingKeywords.isNotEmpty)
                       Column(
                         crossAxisAlignment: CrossAxisAlignment.start,
                         children: [
                           _SectionHeader(
                             title: 'Keywords You\'re Missing',
                             icon: Icons.auto_awesome_outlined,
                           ),
                           Text(
                             'These keywords appear frequently in high-performing ${domain} content',
                             style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                           ),
                           const SizedBox(height: 8),
                           Card(
                             child: Padding(
                               padding: const EdgeInsets.all(16),
                               child: Column(
                                 children: missingKeywords.map((keyword) {
                                   return Padding(
                                     padding: const EdgeInsets.only(bottom: 12),
                                     child: Row(
                                       mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                       children: [
                                         Expanded(
                                           child: Column(
                                             crossAxisAlignment: CrossAxisAlignment.start,
                                             children: [
                                               Text(
                                                 keyword.keyword,
                                                 style: Theme.of(context).textTheme.labelMedium,
                                               ),
                                             ],
                                           ),
                                         ),
                                         Chip(
                                           label: Text('${keyword.score.toStringAsFixed(2)}'),
                                           side: const BorderSide(color: Color(0xFFE0E0E0)),
                                           backgroundColor: Colors.transparent,
                                         ),
                                       ],
                                     ),
                                   );
                                 }).toList(),
                               ),
                             ),
                           ),
                           const SizedBox(height: 24),
                         ],
                       ),

                     // Action Buttons
                     Row(
                       children: [
                         Expanded(
                           child: FilledButton.icon(
                             onPressed: _saveToIdeas,
                             icon: const Icon(Icons.bookmark_outline),
                             label: const Text('Save Idea'),
                           ),
                         ),
                         const SizedBox(width: 12),
                         Expanded(
                           child: OutlinedButton.icon(
                             onPressed: () => Navigator.pushNamed(context, '/dashboard'),
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
          Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                 fontWeight: FontWeight.bold,
                ),
          ),
        ],
      ),
    );
  }
}
