import 'package:flutter/material.dart';

class AppShell extends StatelessWidget {
  const AppShell({
    super.key,
    required this.title,
    required this.child,
    this.actions = const [],
    this.currentRoute,
    this.isAdmin = false,
    this.onLogout,
  });

  final String title;
  final Widget child;
  final List<Widget> actions;
  final String? currentRoute;
  final bool isAdmin;
  final Future<void> Function()? onLogout;

  void _navigate(BuildContext context, String route) {
    if (currentRoute == route) {
      Navigator.pop(context);
      return;
    }
    Navigator.pop(context);
    Navigator.pushNamed(context, route);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title), actions: actions),
      drawer: Drawer(
        child: SafeArea(
          child: ListView(
            children: [
              ListTile(
                leading: const Icon(Icons.dashboard_outlined),
                title: const Text('Dashboard'),
                selected: currentRoute == '/dashboard',
                onTap: () => _navigate(context, '/dashboard'),
              ),
              ListTile(
                leading: const Icon(Icons.upload_file_outlined),
                title: const Text('Analyze My Clip'),
                selected: currentRoute == '/upload',
                onTap: () => _navigate(context, '/upload'),
              ),
              ListTile(
                leading: const Icon(Icons.history_outlined),
                title: const Text('History / My Ideas'),
                selected: currentRoute == '/history',
                onTap: () => _navigate(context, '/history'),
              ),
              if (isAdmin) const Divider(),
              if (isAdmin)
                ListTile(
                  leading: const Icon(Icons.admin_panel_settings_outlined),
                  title: const Text('Admin Console'),
                  selected: currentRoute == '/admin-console',
                  onTap: () => _navigate(context, '/admin-console'),
                ),
              if (isAdmin)
                ListTile(
                  leading: const Icon(Icons.storage_outlined),
                  title: const Text('Admin Datasets'),
                  selected: currentRoute == '/admin-datasets',
                  onTap: () => _navigate(context, '/admin-datasets'),
                ),
              if (isAdmin)
                ListTile(
                  leading: const Icon(Icons.bubble_chart_outlined),
                  title: const Text('Cluster Runs'),
                  selected: currentRoute == '/admin-clusters',
                  onTap: () => _navigate(context, '/admin-clusters'),
                ),
              if (isAdmin)
                ListTile(
                  leading: const Icon(Icons.receipt_long_outlined),
                  title: const Text('System Logs'),
                  selected: currentRoute == '/admin-logs',
                  onTap: () => _navigate(context, '/admin-logs'),
                ),
              if (onLogout != null) const Divider(),
              if (onLogout != null)
                ListTile(
                  leading: const Icon(Icons.logout),
                  title: const Text('Logout'),
                  onTap: () async {
                    Navigator.pop(context);
                    await onLogout!.call();
                  },
                ),
            ],
          ),
        ),
      ),
      body: child,
    );
  }
}
