import 'package:flutter/material.dart';

import 'routing/app_router.dart';
import 'state/auth_controller.dart';
import 'state/auth_scope.dart';
import 'state/theme_controller.dart';

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
      child: ValueListenableBuilder<ThemeMode>(
        valueListenable: themeModeNotifier,
        builder: (context, themeMode, child) {
          return MaterialApp(
            title: 'Content AI',
            debugShowCheckedModeBanner: false,
            theme: _buildLightTheme(),
            darkTheme: _buildDarkTheme(),
            themeMode: themeMode,
            initialRoute: '/',
            onGenerateRoute: _router.onGenerateRoute,
          );
        },
      ),
    );
  }

  ThemeData _buildLightTheme() {
    const primaryColor = Color(0xFF00A8E8);
    const secondaryColor = Color(0xFFFF006E);
    const accentColor = Color(0xFF00F5FF);
    const lightBg = Color(0xFFFAFAFA);
    const cardBg = Color(0xFFFFFFFF);

    return ThemeData(
      colorScheme: ColorScheme.light(
        primary: primaryColor,
        secondary: secondaryColor,
        tertiary: accentColor,
        surface: lightBg,
        surfaceVariant: const Color(0xFFF0F0F0),
      ),
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: lightBg,
      appBarTheme: const AppBarTheme(
        centerTitle: false,
        backgroundColor: cardBg,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        iconTheme: IconThemeData(color: Color(0xFF0A0E27)),
        titleTextStyle: TextStyle(
          color: Color(0xFF0A0E27),
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 1,
        margin: const EdgeInsets.symmetric(vertical: 6),
        color: cardBg,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: const Color(0xFFF0F0F0),
        selectedColor: primaryColor,
        labelStyle: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Color(0xFFE0E0E0)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        filled: true,
        fillColor: const Color(0xFFF9F9F9),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor,
          side: const BorderSide(color: primaryColor),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.w900,
          color: Color(0xFF0A0E27),
        ),
        displayMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w800,
          color: Color(0xFF0A0E27),
        ),
        headlineSmall: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: Color(0xFF0A0E27),
        ),
        titleLarge: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: Color(0xFF0A0E27),
        ),
        titleMedium: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: Color(0xFF0A0E27),
        ),
        labelMedium: TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: Color(0xFF0A0E27),
        ),
        bodyMedium: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: Color(0xFF4A4A4A),
        ),
        bodySmall: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w400,
          color: Color(0xFF7A7A7A),
        ),
      ),
      iconTheme: const IconThemeData(
        color: primaryColor,
        size: 24,
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        circularTrackColor: Color(0xFFE0E0E0),
        linearTrackColor: Color(0xFFE0E0E0),
      ),
    );
  }

  ThemeData _buildDarkTheme() {
    const primaryColor = Color(0xFF00A8E8);
    const secondaryColor = Color(0xFFFF006E);
    const accentColor = Color(0xFF00F5FF);
    const darkBg = Color(0xFF0A0E27);
    const cardBg = Color(0xFF111736);

    return ThemeData(
      colorScheme: ColorScheme.dark(
        primary: primaryColor,
        secondary: secondaryColor,
        tertiary: accentColor,
        surface: cardBg,
        surfaceVariant: const Color(0xFF131A3A),
        background: darkBg,
      ),
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: darkBg,
      appBarTheme: const AppBarTheme(
        centerTitle: false,
        backgroundColor: Color(0xFF111736),
        elevation: 0,
        surfaceTintColor: Colors.transparent,
        iconTheme: IconThemeData(color: Colors.white),
        titleTextStyle: TextStyle(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.bold,
        ),
      ),
      cardTheme: CardThemeData(
        elevation: 1,
        margin: const EdgeInsets.symmetric(vertical: 6),
        color: cardBg,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: const Color(0xFF1A224D),
        selectedColor: primaryColor,
        labelStyle: const TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Color(0xFF33417F)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: primaryColor, width: 2),
        ),
        filled: true,
        fillColor: const Color(0xFF142051),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor,
          side: const BorderSide(color: primaryColor),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.w900,
          color: Colors.white,
        ),
        displayMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.w800,
          color: Colors.white,
        ),
        headlineSmall: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: Colors.white,
        ),
        titleLarge: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: Colors.white,
        ),
        titleMedium: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        labelMedium: TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        bodyMedium: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: Color(0xFFECECEC),
        ),
        bodySmall: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w400,
          color: Color(0xFFA8B0D3),
        ),
      ),
      iconTheme: const IconThemeData(
        color: primaryColor,
        size: 24,
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        circularTrackColor: Color(0xFF33417F),
        linearTrackColor: Color(0xFF33417F),
      ),
    );
  }
}
