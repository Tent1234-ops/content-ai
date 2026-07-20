import 'package:flutter/material.dart';

import '../state/auth_scope.dart';

class AuthGate extends StatelessWidget {
  const AuthGate({super.key});

  @override
  Widget build(BuildContext context) {
    final auth = AuthScope.of(context);
    if (!auth.initialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!context.mounted) return;
      Navigator.pushReplacementNamed(
        context,
        auth.isAuthenticated ? '/dashboard' : '/login',
      );
    });

    return const Scaffold(body: Center(child: CircularProgressIndicator()));
  }
}
