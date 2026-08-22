class NotebookLmMarkdownDocument {
  const NotebookLmMarkdownDocument({
    required this.transcript,
    this.sourceTitle,
    this.creatorChannel,
    this.sourceUrl,
  });

  final String transcript;
  final String? sourceTitle;
  final String? creatorChannel;
  final String? sourceUrl;
}

class NotebookLmMarkdownParser {
  static const minTranscriptCharacters = 80;
  static const maxTranscriptCharacters = 2000000;

  static final RegExp _transcriptHeading = RegExp(
    r'^(#{1,6})\s*(cleaned transcription text|cleaned transcript|full source transcript|full transcript|transcription text|transcript|ข้อความถอดเสียง|บทถอดเสียง|ทรานสคริปต์)\s*$',
    caseSensitive: false,
  );
  static final RegExp _anyHeading = RegExp(r'^(#{1,6})\s+\S');
  static final RegExp _boldMetadata = RegExp(
    r'^\s*[-*]\s+\*\*([^*]+?):\*\*\s*(.*?)\s*$',
    caseSensitive: false,
  );
  static final RegExp _plainMetadata = RegExp(
    r'^\s*[-*]\s+([^:]+):\s*(.*?)\s*$',
    caseSensitive: false,
  );

  static NotebookLmMarkdownDocument parse(String markdown) {
    final normalized = markdown
        .replaceFirst('\uFEFF', '')
        .replaceAll('\r\n', '\n')
        .replaceAll('\r', '\n');
    if (normalized.trim().isEmpty) {
      throw const FormatException('The Markdown file is empty.');
    }

    final lines = normalized.split('\n');
    final metadata = _readMetadata(lines);
    var headingIndex = -1;
    var headingLevel = 0;
    for (var index = 0; index < lines.length; index++) {
      final match = _transcriptHeading.firstMatch(lines[index].trim());
      if (match == null) continue;
      headingIndex = index;
      headingLevel = (match.group(1) ?? '').length;
      break;
    }
    if (headingIndex < 0) {
      throw const FormatException(
        'Transcript section not found. Expected a heading such as '
        '"## Cleaned Transcription Text".',
      );
    }

    var endIndex = lines.length;
    for (var index = headingIndex + 1; index < lines.length; index++) {
      final match = _anyHeading.firstMatch(lines[index].trim());
      final level = (match?.group(1) ?? '').length;
      if (match != null && level <= headingLevel) {
        endIndex = index;
        break;
      }
    }

    final transcriptLines = lines.sublist(headingIndex + 1, endIndex);
    _trimBlankAndRuleLines(transcriptLines);
    if (transcriptLines.isNotEmpty &&
        transcriptLines.first.trimLeft().startsWith('```')) {
      transcriptLines.removeAt(0);
    }
    if (transcriptLines.isNotEmpty &&
        transcriptLines.last.trimLeft().startsWith('```')) {
      transcriptLines.removeLast();
    }
    _trimBlankAndRuleLines(transcriptLines);

    final transcript = transcriptLines.join('\n').trim();
    if (transcript.length < minTranscriptCharacters) {
      throw const FormatException(
        'The transcript section must contain at least 80 characters.',
      );
    }
    if (transcript.length > maxTranscriptCharacters) {
      throw const FormatException(
        'The transcript exceeds the 2,000,000 character limit.',
      );
    }

    return NotebookLmMarkdownDocument(
      transcript: transcript,
      sourceTitle: metadata['source video'] ?? metadata['video title'],
      creatorChannel: metadata['creator channel'] ?? metadata['channel title'],
      sourceUrl: metadata['source url'] ??
          metadata['video url'] ??
          metadata['youtube url'] ??
          _urlFrom(metadata['source video']),
    );
  }

  static Map<String, String> _readMetadata(List<String> lines) {
    final values = <String, String>{};
    for (final line in lines) {
      final match =
          _boldMetadata.firstMatch(line) ?? _plainMetadata.firstMatch(line);
      if (match == null) continue;
      final key = (match.group(1) ?? '').trim().toLowerCase();
      final value = (match.group(2) ?? '').trim();
      if (key.isNotEmpty && value.isNotEmpty) values[key] = value;
    }
    return values;
  }

  static String? _urlFrom(String? value) {
    if (value == null) return null;
    final match = RegExp(r'https?://[^\s)>]+').firstMatch(value);
    return match?.group(0);
  }

  static void _trimBlankAndRuleLines(List<String> lines) {
    while (lines.isNotEmpty && _isBlankOrRule(lines.first)) {
      lines.removeAt(0);
    }
    while (lines.isNotEmpty && _isBlankOrRule(lines.last)) {
      lines.removeLast();
    }
  }

  static bool _isBlankOrRule(String value) {
    final trimmed = value.trim();
    return trimmed.isEmpty ||
        RegExp(r'^(-{3,}|\*{3,}|_{3,})$').hasMatch(trimmed);
  }
}
