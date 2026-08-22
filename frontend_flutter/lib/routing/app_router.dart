import 'package:flutter/material.dart';

import '../screens/admin_cluster_runs_screen.dart';
import '../screens/auth_gate.dart';
import '../screens/admin_console_screen.dart';
import '../screens/admin_dataset_review_screen.dart';
import '../screens/admin_datasets_screen.dart';
import '../screens/admin_logs_screen.dart';
import '../screens/admin_transcript_import_screen.dart';
import '../screens/dashboard_screen.dart';
import '../screens/history_screen.dart';
import '../screens/login_screen.dart';
import '../screens/register_screen.dart';
import '../screens/result_screen.dart';
import '../screens/upload_screen.dart';
import '../state/auth_controller.dart';

class AppRouter {
  AppRouter(this.authController);

  final AuthController authController;

  static const Set<String> _publicRoutes = {'/', '/login', '/register'};
  static const Set<String> _adminRoutes = {
    '/admin-console',
    '/admin-dataset-review',
    '/admin-datasets',
    '/admin-clusters',
    '/admin-logs',
    '/admin-transcript-import',
  };

  Route<dynamic> onGenerateRoute(RouteSettings settings) {
    final name = settings.name ?? '/';
    if (!authController.initialized) {
      return _page(const AuthGate(), settings);
    }
    if (!_publicRoutes.contains(name) && !authController.isAuthenticated) {
      return _page(const LoginScreen(), settings.copyWith(name: '/login'));
    }
    if (_adminRoutes.contains(name) && !authController.isAdmin) {
      return _page(
          const DashboardScreen(), settings.copyWith(name: '/dashboard'));
    }

    switch (name) {
      case '/':
        return _page(const AuthGate(), settings);
      case '/login':
        return _page(const LoginScreen(), settings);
      case '/register':
        return _page(const RegisterScreen(), settings);
      case '/dashboard':
        return _page(const DashboardScreen(), settings);
      case '/upload':
        return _page(const UploadScreen(), settings);
      case '/history':
        return _page(const HistoryScreen(), settings);
      case '/result':
        return _page(const ResultScreen(), settings);
      case '/admin-console':
        return _page(const AdminConsoleScreen(), settings);
      case '/admin-dataset-review':
        return _page(const AdminDatasetReviewScreen(), settings);
      case '/admin-datasets':
        return _page(const AdminDatasetsScreen(), settings);
      case '/admin-clusters':
        return _page(const AdminClusterRunsScreen(), settings);
      case '/admin-logs':
        return _page(const AdminLogsScreen(), settings);
      case '/admin-transcript-import':
        return _page(const AdminTranscriptImportScreen(), settings);
      default:
        return _page(
            const DashboardScreen(), settings.copyWith(name: '/dashboard'));
    }
  }

  MaterialPageRoute<dynamic> _page(Widget child, RouteSettings settings) {
    return MaterialPageRoute<dynamic>(
        builder: (_) => child, settings: settings);
  }
}

extension on RouteSettings {
  RouteSettings copyWith({String? name, Object? arguments}) {
    return RouteSettings(
        name: name ?? this.name, arguments: arguments ?? this.arguments);
  }
}
