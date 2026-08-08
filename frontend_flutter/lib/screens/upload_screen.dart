import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/recommendation_result.dart';
import '../repositories/analysis_repository.dart';
import '../state/auth_scope.dart';
import '../widgets/app_shell.dart';
import '../widgets/state_widgets.dart';
import 'result_screen.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  final _repository = AnalysisRepository();
  String? _error;
  bool _loading = false;
  String? _selectedFileName;
  String? _selectedFilePath;
  Uint8List? _selectedFileBytes;
  Stream<List<int>>? _selectedFileStream;
  int? _selectedFileSize;
  int _uploadProgress = 0;
  String _statusMessage = '';
  Timer? _progressTimer;
  String? _suggestedTopic;
  bool _hasReadRouteArgs = false;

  Future<void> _pickFile() async {
    try {
      final file = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
        withData: true,
      );

      if (file == null) {
        return;
      }

      final selectedFile = file.files.single;
      final selectedBytes = selectedFile.bytes;
      final selectedStream = selectedFile.readStream;
      String? selectedPath;
      if (!kIsWeb) {
        try {
          selectedPath = selectedFile.path;
        } catch (_) {
          selectedPath = null;
        }
      }

      if (selectedPath == null &&
          selectedBytes == null &&
          selectedStream == null) {
        setState(() {
          _error =
              'Cannot read selected file on this platform. Please try a different browser or device.';
        });
        return;
      }

      final fileSizeMB = (selectedFile.size / (1024 * 1024)).toStringAsFixed(1);

      setState(() {
        _selectedFileName = selectedFile.name;
        _selectedFilePath = selectedPath;
        _selectedFileBytes = selectedBytes;
        _selectedFileStream = selectedStream;
        _selectedFileSize = selectedFile.size;
        _error = null;
        _uploadProgress = 0;
        _statusMessage = 'File ready ($fileSizeMB MB)';
      });

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Selected: ${selectedFile.name}'),
          duration: const Duration(seconds: 2),
        ),
      );
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Error picking file: ${e.toString()}');
      }
    }
  }

  Future<void> _uploadAndAnalyze() async {
    if (_selectedFileName == null ||
        (_selectedFilePath == null &&
            _selectedFileBytes == null &&
            _selectedFileStream == null)) {
      setState(() => _error = 'Please select a file first');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _uploadProgress = 5;
      _statusMessage = 'Uploading video...';
    });

    _startProgressTimer();
    try {
      final jobId = await _repository.startAnalyzeAndSaveVideo(
        fileName: _selectedFileName!,
        filePath: _selectedFilePath,
        fileBytes: _selectedFileBytes,
        fileStream: _selectedFileStream,
        fileSize: _selectedFileSize,
      );

      if (!mounted) return;
      setState(() {
        _uploadProgress = 20;
        _statusMessage = 'Queued job $jobId';
      });

      final response = await _pollAnalysisJob(jobId);

      if (!mounted) return;

      _progressTimer?.cancel();
      setState(() {
        _uploadProgress = 100;
        _statusMessage = 'Analysis complete!';
      });

      await Future.delayed(const Duration(milliseconds: 500));

      if (!mounted) return;

      Navigator.pushReplacementNamed(
        context,
        '/result',
        arguments: ResultScreenArgs(initialData: response),
      );
    } catch (error) {
      _progressTimer?.cancel();
      if (!mounted) return;
      setState(() {
        _error = error.toString();
        _statusMessage = 'Error: ${error.toString()}';
      });
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<AnalysisResultViewData> _pollAnalysisJob(String jobId) async {
    for (var attempt = 0; attempt < 600; attempt++) {
      final job = await _repository.getAnalysisJob(jobId);
      if (!mounted) {
        throw Exception('Upload screen was closed.');
      }

      setState(() {
        if (job.status == 'queued') {
          _uploadProgress = _uploadProgress < 30 ? 30 : _uploadProgress;
          _statusMessage = 'Queued for analysis...';
        } else if (job.status == 'running') {
          _uploadProgress = _uploadProgress < 85 ? 85 : _uploadProgress;
          _statusMessage = 'Analyzing video...';
        } else {
          _statusMessage = 'Job status: ${job.status}';
        }
      });

      if (job.isComplete) {
        final result = job.result;
        if (result == null) {
          throw Exception('Analysis completed but no result was returned.');
        }
        return result;
      }

      if (job.isFailed) {
        throw Exception(job.error ?? 'Analysis failed.');
      }

      await Future.delayed(const Duration(seconds: 2));
    }
    throw Exception('Analysis is still running. Please check History later or try a shorter clip.');
  }

  void _startProgressTimer() {
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(milliseconds: 800), (timer) {
      if (!mounted || !_loading) {
        timer.cancel();
        return;
      }

      setState(() {
        if (_uploadProgress < 25) {
          _uploadProgress = 25;
          _statusMessage = 'Extracting audio...';
        } else if (_uploadProgress < 50) {
          _uploadProgress = 50;
          _statusMessage = 'Generating transcript...';
        } else if (_uploadProgress < 75) {
          _uploadProgress = 75;
          _statusMessage = 'Classifying content...';
        } else if (_uploadProgress < 90) {
          _uploadProgress = 90;
          _statusMessage = 'Processing AI analysis...';
        } else if (_uploadProgress < 92) {
          _uploadProgress = 92;
          _statusMessage = 'Finishing up report...';
        } else if (_uploadProgress < 94) {
          _uploadProgress = 94;
          _statusMessage = 'Still processing analysis...';
        }
      });
    });
  }

  void _clearSelection() {
    _progressTimer?.cancel();
    setState(() {
      _selectedFileName = null;
      _selectedFilePath = null;
      _selectedFileBytes = null;
      _uploadProgress = 0;
      _statusMessage = '';
      _error = null;
    });
  }

  @override
  void didChangeDependencies() {
   super.didChangeDependencies();
   if (!_hasReadRouteArgs) {
     final args = ModalRoute.of(context)?.settings.arguments;
     if (args is Map<String, dynamic>) {
       _suggestedTopic = args['suggestedTopic']?.toString();
     }
     _hasReadRouteArgs = true;
   }
  }

  @override
  void dispose() {
   _progressTimer?.cancel();
   super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    final hasFile = _selectedFileName != null;

    return AppShell(
      title: 'Analyze My Clip',
      currentRoute: '/upload',
      isAdmin: auth.isAdmin,
      child: RefreshIndicator(
        onRefresh: () async {
          _clearSelection();
        },
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: ListView(
            children: [
              // Header Card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.video_camera_back_outlined,
                            size: 28,
                            color: Theme.of(context).primaryColor,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Analyze Your Clip',
                                  style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 18,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'Upload a video to get content analysis',
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      const Divider(height: 1),
                      const SizedBox(height: 12),
                      const Text(
                        'What we analyze:',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 8),
                      _AnalysisFeature(
                        icon: Icons.subtitles_outlined,
                        title: 'Transcript',
                        description: 'Auto-generated from your video',
                      ),
                      const SizedBox(height: 8),
                      _AnalysisFeature(
                        icon: Icons.category_outlined,
                        title: 'Content Classification',
                        description: 'AI-predicted domain/category',
                      ),
                      const SizedBox(height: 8),
                      _AnalysisFeature(
                        icon: Icons.key_outlined,
                        title: 'Keywords & Gaps',
                        description:
                            'Missing keywords compared to top performers',
                      ),
                      const SizedBox(height: 8),
                      _AnalysisFeature(
                        icon: Icons.lightbulb_outline,
                        title: 'Hook Suggestions',
                        description: 'Recommended first 60 seconds keywords',
                      ),
                      const SizedBox(height: 8),
                      _AnalysisFeature(
                        icon: Icons.schedule_outlined,
                        title: 'Duration Recommendation',
                        description: 'Optimal video length for your domain',
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              if (_suggestedTopic != null)
                Card(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Suggested trend to analyze',
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Upload a clip related to "$_suggestedTopic" to get recommendations that match this trend.',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      ],
                    ),
                  ),
                ),
              if (_suggestedTopic != null) const SizedBox(height: 16),
 
              // Upload Section
              if (!hasFile) ...[
                FilledButton.icon(
                  onPressed: _loading ? null : _pickFile,
                  icon: const Icon(Icons.upload_file),
                  label: const Text('Pick a Video File'),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: _loading ? null : _pickFile,
                  icon: const Icon(Icons.cloud_upload_outlined),
                  label: const Text('Or tap to browse'),
                ),
              ] else ...[
                // Selected File Card
                Card(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Icon(Icons.check_circle),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    _selectedFileName!,
                                    style:
                                        Theme.of(context).textTheme.titleSmall,
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    _statusMessage,
                                    style:
                                        Theme.of(context).textTheme.bodySmall,
                                  ),
                                ],
                              ),
                            ),
                            if (!_loading)
                              IconButton(
                                icon: const Icon(Icons.close),
                                onPressed: _clearSelection,
                                tooltip: 'Clear selection',
                              ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Upload Progress
                if (_loading) ...[
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _statusMessage,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: LinearProgressIndicator(
                          value: _uploadProgress / 100,
                          minHeight: 8,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${_uploadProgress}% complete',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                ] else ...[
                  FilledButton.icon(
                    onPressed: _uploadAndAnalyze,
                    icon: const Icon(Icons.analytics_outlined),
                    label: const Text('Analyze Now'),
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton(
                    onPressed: _clearSelection,
                    child: const Text('Choose Different File'),
                  ),
                ],
              ],

              const SizedBox(height: 16),

              // Error State
              if (_error != null)
                ErrorStateView(
                  message: _error!,
                  onRetry: hasFile ? _uploadAndAnalyze : _pickFile,
                ),

              const SizedBox(height: 16),

              // Tips Card
              if (!_loading)
                Card(
                  color: Theme.of(context).colorScheme.surfaceVariant,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.info_outline,
                              size: 20,
                              color: Theme.of(context)
                                  .colorScheme
                                  .onSurfaceVariant,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Tips for better results',
                              style: Theme.of(context).textTheme.labelLarge,
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        _TipRow(
                            text: 'Use clear audio for better transcription'),
                        const SizedBox(height: 8),
                        _TipRow(
                            text:
                                'Keep videos between 30 seconds to 10 minutes'),
                        const SizedBox(height: 8),
                        _TipRow(text: 'MP4, WebM, or MOV formats work best'),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _AnalysisFeature extends StatelessWidget {
  const _AnalysisFeature({
    required this.icon,
    required this.title,
    required this.description,
  });

  final IconData icon;
  final String title;
  final String description;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: Theme.of(context).primaryColor),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.labelMedium,
              ),
              const SizedBox(height: 2),
              Text(
                description,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _TipRow extends StatelessWidget {
  const _TipRow({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 3),
          child: Icon(
            Icons.check,
            size: 18,
            color: Theme.of(context).colorScheme.primary,
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ),
      ],
    );
  }
}
