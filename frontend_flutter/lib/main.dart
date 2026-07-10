import 'package:flutter/material.dart';

import 'routing/app_router.dart';
import 'state/auth_controller.dart';
import 'state/auth_scope.dart';

void main() {
  runApp(const ContentAiApp());
}

class ContentAiApp extends StatefulWidget {
  const ContentAiApp({super.key});

  @override
  State<ContentAiApp> createState() => _ContentAiAppState();
}

class _ContentAiAppState extends State<ContentAiApp> {
  late final AuthController _authController;
  late final AppRouter _router;

  @override
  void initState() {
    super.initState();
    _authController = AuthController();
    _router = AppRouter(_authController);
    _authController.initialize();
  }

  @override
  void dispose() {
    _authController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AuthScope(
      controller: _authController,
      child: MaterialApp(
        title: 'Content AI',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0E7490)),
          useMaterial3: true,
          scaffoldBackgroundColor: const Color(0xFFF6FAFC),
          appBarTheme: const AppBarTheme(centerTitle: false),
          cardTheme: const CardThemeData(
            elevation: 0,
            margin: EdgeInsets.symmetric(vertical: 6),
          ),
          inputDecorationTheme: const InputDecorationTheme(
            border: OutlineInputBorder(),
          ),
        ),
        initialRoute: '/',
        onGenerateRoute: _router.onGenerateRoute,
      ),
    );
  }
}
