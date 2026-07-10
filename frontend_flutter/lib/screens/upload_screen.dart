import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

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

  Future<void> _pickAndUpload() async {
    final file = await FilePicker.platform.pickFiles(type: FileType.video);
    if (file == null || file.files.single.path == null) {
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
      _selectedFileName = file.files.single.name;
    });
    try {
      final response =
          await _repository.analyzeAndSaveVideo(file.files.single.path!);
      if (!mounted) return;
      Navigator.pushReplacementNamed(
        context,
        '/result',
        arguments: ResultScreenArgs(initialData: response),
      );
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
      title: 'Analyze My Clip',
      currentRoute: '/upload',
      isAdmin: auth.isAdmin,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Upload a clip',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 18)),
                    SizedBox(height: 8),
                    Text(
                      'Pick an MP4 or video file and the backend will generate transcript, keywords, '
                      'gap analysis, hook suggestions and recommended duration.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _loading ? null : _pickAndUpload,
              icon: const Icon(Icons.upload_file),
              label: Text(_loading ? 'Uploading...' : 'Pick video and analyze'),
            ),
            if (_selectedFileName != null) ...[
              const SizedBox(height: 12),
              Text('Selected: $_selectedFileName'),
            ],
            const SizedBox(height: 16),
            if (_loading) const LinearProgressIndicator(),
            if (_error != null)
              ErrorStateView(
                message: _error!,
                onRetry: _pickAndUpload,
              ),
          ],
        ),
      ),
    );
  }
}
