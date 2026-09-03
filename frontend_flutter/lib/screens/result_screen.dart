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
    final contentKeywords = recommendation?.contentKeywords ?? const <String>[];
    final comparableKeywords =
        recommendation?.comparableKeywords ?? const <String>[];
    final userKeywords = comparableKeywords.isNotEmpty
        ? comparableKeywords
        : recommendation?.userKeywords ?? const <String>[];
    final hookTerms = recommendation?.hookTerms ?? const <String>[];
    final missingKeywords =
        recommendation?.missingKeywords ?? const <KeywordScore>[];
    final hookKeywords = recommendation?.hookKeywords ?? const <KeywordScore>[];
    final duration = recommendation?.duration;
    final evidence = recommendation?.evidence;
    final hasReferenceEvidence =
        (recommendation?.datasetProfile.sampleSize ?? 0) > 0;
    final missingKeywordsEmptyMessage = hasReferenceEvidence
        ? 'ยังไม่พบหัวข้อเพิ่มเติมที่มีหลักฐานสนับสนุนเพียงพอจากคลิปอ้างอิง'
        : 'ยังไม่มีข้อมูลคลิปอ้างอิงในหมวดนี้เพียงพอสำหรับสร้างคำแนะนำ';
    final hookKeywordsEmptyMessage = hasReferenceEvidence
        ? 'ยังไม่พบคำแนะนำเพิ่มเติมสำหรับช่วงเปิดคลิปจากข้อมูลอ้างอิง'
        : 'ยังไม่มีข้อมูลคลิปอ้างอิงในหมวดนี้เพียงพอสำหรับแนะนำช่วงเปิดคลิป';
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
                            label: Text('หมวดหมู่: $domain'),
                          ),
                          Chip(
                            avatar: const Icon(Icons.check_circle, size: 18),
                            label: Text(
                              'ความมั่นใจ ${classifierConfidence.toStringAsFixed(0)}%',
                            ),
                          ),
                          if (isSaved)
                            const Chip(
                              avatar:
                                  Icon(Icons.bookmark_added_outlined, size: 18),
                              label: Text('บันทึกในไอเดียของฉันแล้ว'),
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
                        hasReferenceEvidence: hasReferenceEvidence,
                      ),
                      const SizedBox(height: 24),
                      if (data.transcript.isNotEmpty) ...[
                        const _SectionHeader(
                          title: 'ตัวอย่างข้อความถอดเสียง',
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
                        title: 'หมวดหมู่ของคลิป',
                        icon: Icons.account_tree_outlined,
                      ),
                      _ClassificationCard(
                        domain: domain,
                        confidence: classifierConfidence,
                        classification: classification,
                      ),
                      const SizedBox(height: 24),
                      const _SectionHeader(
                        title: 'คำสำคัญที่พบทั้งคลิป',
                        icon: Icons.article_outlined,
                      ),
                      _StringKeywordCard(
                        keywords: contentKeywords,
                        emptyMessage: 'ยังไม่พบคำสำคัญจากเนื้อหาในคลิปนี้',
                      ),
                      const SizedBox(height: 24),
                      const _SectionHeader(
                        title: 'หัวข้อหลักที่ใช้เปรียบเทียบ',
                        icon: Icons.compare_arrows_outlined,
                      ),
                      _StringKeywordCard(
                        keywords: comparableKeywords,
                        emptyMessage:
                            'ยังไม่พบหัวข้อที่ระบบรู้จักสำหรับใช้เปรียบเทียบ',
                      ),
                      const SizedBox(height: 24),
                      const _SectionHeader(
                        title: 'คำสำคัญที่พบในช่วงเปิดคลิป',
                        icon: Icons.bolt_outlined,
                      ),
                      _StringKeywordCard(
                        keywords: hookTerms,
                        emptyMessage:
                            'ยังไม่พบคำสำคัญจากเสียงพูดในช่วงเปิดคลิป',
                      ),
                      const SizedBox(height: 24),
                      const _SectionHeader(
                        title: 'หัวข้อที่ควรเพิ่มในคลิป',
                        icon: Icons.auto_awesome_outlined,
                      ),
                      if (missingKeywords.isNotEmpty) ...[
                        Text(
                          'หัวข้อเหล่านี้พบในคลิปตัวอย่างหมวด $domain ที่มีผลตอบรับสูง '
                          'แต่ยังไม่พบในเนื้อหาของคุณหรือคำที่มีความหมายใกล้กัน',
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: Colors.grey),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          'เรียงลำดับจากจำนวนคลิปอ้างอิงที่พูดถึง ความถี่ '
                          'และผลตอบรับของคลิป',
                        ),
                        const SizedBox(height: 8),
                      ],
                      _ScoredKeywordCard(
                        keywords: missingKeywords,
                        emptyMessage: missingKeywordsEmptyMessage,
                      ),
                      const SizedBox(height: 24),
                      if (duration != null) ...[
                        const _SectionHeader(
                          title: 'ความยาวคลิปที่แนะนำ',
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
                      const _SectionHeader(
                        title: 'คำแนะนำสำหรับช่วงเปิดคลิป',
                        icon: Icons.lightbulb_outline,
                      ),
                      _ScoredKeywordCard(
                        keywords: hookKeywords,
                        emptyMessage: hookKeywordsEmptyMessage,
                      ),
                      const SizedBox(height: 24),
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
    required this.hasReferenceEvidence,
  });

  final String domain;
  final double confidence;
  final List<String> userKeywords;
  final List<KeywordScore> missingKeywords;
  final DurationRecommendation? duration;
  final bool hasReferenceEvidence;

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
              label: 'หมวดหมู่คลิป',
              value: '$domain (${confidence.toStringAsFixed(0)}%)',
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.key_outlined,
              label: 'หัวข้อที่พบในคลิป',
              value:
                  userKeywords.isEmpty ? '-' : userKeywords.take(6).join(', '),
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.auto_awesome_outlined,
              label: 'หัวข้อที่ควรเพิ่ม',
              value: missingKeywords.isEmpty
                  ? hasReferenceEvidence
                      ? 'ยังไม่พบหัวข้อเพิ่มเติมที่มีหลักฐานเพียงพอ'
                      : 'ข้อมูลอ้างอิงยังไม่เพียงพอ'
                  : missingKeywords
                      .take(6)
                      .map((item) => item.keyword)
                      .join(', '),
            ),
            const Divider(height: 20),
            _SummaryRow(
              icon: Icons.schedule_outlined,
              label: 'ความยาวที่แนะนำ',
              value: duration?.recommendedRange ?? '-',
            ),
          ],
        ),
      ),
    );
  }
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
                        'หมวดหมู่ที่โมเดลคาดการณ์',
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
                      'ความมั่นใจ',
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
                'หมวดหมู่อื่นที่เป็นไปได้',
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
  const _ScoredKeywordCard({
    required this.keywords,
    required this.emptyMessage,
  });

  final List<KeywordScore> keywords;
  final String emptyMessage;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: keywords.isEmpty
            ? Text(emptyMessage)
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  for (var index = 0; index < keywords.length; index++) ...[
                    if (index > 0) const Divider(height: 24),
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      keywords[index].keyword,
                                      style: Theme.of(context)
                                          .textTheme
                                          .titleSmall,
                                    ),
                                    if (keywords[index].hasDatasetEvidence) ...[
                                      const SizedBox(height: 4),
                                      Text(
                                        'พบหัวข้อนี้ในคลิปตัวอย่างที่มีผลตอบรับสูง '
                                        '${keywords[index].supportCount} จาก ${keywords[index].sampleSize} คลิป '
                                        '(กล่าวถึงรวม ${keywords[index].totalFrequency} ครั้ง)',
                                        style: Theme.of(context)
                                            .textTheme
                                            .bodySmall,
                                      ),
                                    ],
                                  ],
                                ),
                              ),
                              const SizedBox(width: 12),
                              Chip(
                                label: Text(
                                  keywords[index].hasDatasetEvidence
                                      ? 'คะแนนสนับสนุน ${(keywords[index].score.clamp(0.0, 1.0) * 100).toStringAsFixed(0)}%'
                                      : keywords[index]
                                          .score
                                          .toStringAsFixed(2),
                                ),
                                side:
                                    const BorderSide(color: Color(0xFFE0E0E0)),
                                backgroundColor: Colors.transparent,
                              ),
                            ],
                          ),
                          if (keywords[index]
                              .supportingExamples
                              .isNotEmpty) ...[
                            const SizedBox(height: 10),
                            Text(
                              'ตัวอย่างคลิปอ้างอิง',
                              style: Theme.of(context).textTheme.labelSmall,
                            ),
                            const SizedBox(height: 6),
                            for (final example
                                in keywords[index].supportingExamples) ...[
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Padding(
                                    padding: EdgeInsets.only(top: 2),
                                    child: Icon(Icons.ondemand_video_outlined,
                                        size: 16),
                                  ),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      '${example.title} '
                                      '(ข้อมูล #${example.datasetId}, กล่าวถึง ${example.frequency} ครั้ง)',
                                      style:
                                          Theme.of(context).textTheme.bodySmall,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 4),
                            ],
                          ],
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

class _DurationCard extends StatelessWidget {
  const _DurationCard({required this.duration});

  final DurationRecommendation duration;

  @override
  Widget build(BuildContext context) {
    final isSufficient = duration.hasSufficientEvidence;
    final median = duration.medianSeconds ?? duration.recommendedSeconds;
    final headline = isSufficient && median != null
        ? 'ค่ากลาง $median วินาที'
        : 'ข้อมูลอ้างอิงยังไม่เพียงพอ';
    final detail = isSufficient
        ? 'ช่วงกลางของข้อมูลอ้างอิง: ${duration.recommendedRange} '
            '(เปอร์เซ็นไทล์ ${duration.percentileLow}-${duration.percentileHigh})'
        : 'ขณะนี้มีข้อมูลความยาว ${duration.sampleSize} คลิป '
            'จากขั้นต่ำ ${duration.minimumSampleSize} คลิป';
    return Card(
      color: Theme.of(context).colorScheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  isSufficient ? Icons.analytics_outlined : Icons.info_outline,
                  color: isSufficient
                      ? Theme.of(context).colorScheme.primary
                      : Theme.of(context).colorScheme.error,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        headline,
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                      const SizedBox(height: 4),
                      Text(detail),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 32,
              runSpacing: 12,
              children: [
                _DurationFact(
                  label: 'แหล่งข้อมูล',
                  value: _durationSourceLabel(duration.source),
                ),
                _DurationFact(
                  label: 'จำนวนตัวอย่าง',
                  value: '${duration.sampleSize} คลิป '
                      '(เป้าหมาย ${duration.targetSampleSize})',
                ),
                _DurationFact(
                  label: 'กลุ่มเปรียบเทียบ',
                  value: _durationCohortLabel(duration.cohort),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _DurationFact extends StatelessWidget {
  const _DurationFact({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: const BoxConstraints(minWidth: 150, maxWidth: 280),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.labelSmall),
          Text(value, style: Theme.of(context).textTheme.labelMedium),
        ],
      ),
    );
  }
}

String _durationSourceLabel(String source) {
  if (source == 'youtube_metadata') return 'ข้อมูลความยาวจาก YouTube';
  if (source == 'none') return 'ยังไม่มีแหล่งข้อมูลที่ตรวจสอบแล้ว';
  return source.replaceAll('_', ' ');
}

String _durationCohortLabel(String cohort) {
  if (cohort == 'upload_compatible_under_5m') {
    return 'คลิปอ้างอิงที่ยาวไม่เกิน 5 นาที';
  }
  return cohort.replaceAll('_', ' ');
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
