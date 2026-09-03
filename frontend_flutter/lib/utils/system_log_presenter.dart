String systemLogActionLabel(String action) {
  switch (action.trim().toLowerCase()) {
    case 'notebooklm_candidate_created':
      return 'นำ Transcript เข้าคิวตรวจสอบ';
    case 'dataset_review_approve':
      return 'อนุมัติข้อมูลฝึกเข้าสู่ฐานข้อมูล';
    case 'dataset_review_reject':
      return 'ปฏิเสธข้อมูลฝึกก่อนนำเข้าฐานข้อมูล';
    case 'admin_dataset_create':
      return 'เพิ่มข้อมูล Dataset โดยผู้ดูแล';
    case 'admin_dataset_update':
      return 'แก้ไขข้อมูล Dataset';
    case 'admin_dataset_training_content_corrected':
      return 'แก้ไข Transcript หรือหมวดข้อมูลฝึกที่อนุมัติแล้ว';
    case 'admin_settings_update':
      return 'แก้ไขการตั้งค่าระบบ';
    case 'admin_settings_reset':
      return 'คืนค่าตั้งต้นของระบบ';
    case 'admin_settings_backup':
      return 'สำรองการตั้งค่าระบบ';
    case 'admin_settings_restore':
      return 'กู้คืนการตั้งค่าระบบ';
    case 'video_analyze_save':
      return 'บันทึกผลวิเคราะห์วิดีโอ';
    case 'nlp_extract_save':
      return 'บันทึกผลสกัดคำสำคัญ';
    case 'classification_model_smoke_test':
      return 'ทดสอบขั้นตอนฝึกโมเดลเบื้องต้น';
    case 'classification_model_benchmark':
      return 'ฝึกและประเมินโมเดลจำแนกหมวด';
    case 'classification_model_activate':
      return 'เปิดใช้งานโมเดลจำแนกหมวด';
    case 'youtube_trends_sync':
      return 'อัปเดตข้อมูลเทรนด์ YouTube';
    case 'google_trends_sync':
      return 'อัปเดตข้อมูลเทรนด์ Google';
    case 'tiktok_trends_sync':
      return 'อัปเดตข้อมูลเทรนด์ TikTok';
    case 'trending_fetcher_loop':
      return 'ตัวเก็บเทรนด์รอบหลักทำงานผิดพลาด';
    case 'youtube_category_trend_fetcher_loop':
      return 'ตัวเก็บเทรนด์ YouTube รายหมวดทำงานผิดพลาด';
    case 'purge_legacy_demo_dataset':
      return 'ล้างข้อมูลตัวอย่างรุ่นเก่า';
    case 'kmeans_save':
      return 'บันทึกผลจัดกลุ่ม K-means';
    case 'hdbscan_save':
      return 'บันทึกผลจัดกลุ่ม HDBSCAN';
  }

  final normalized = action.trim().toLowerCase();
  if (normalized.startsWith('trending_fetch_')) {
    final source = normalized.substring('trending_fetch_'.length);
    return 'ดึงข้อมูลเทรนด์ $source ไม่สำเร็จ';
  }
  return action.trim().isEmpty ? 'เหตุการณ์ที่ไม่ระบุชื่อ' : action;
}

String systemLogStatusLabel(String status) {
  switch (status.trim().toLowerCase()) {
    case 'success':
      return 'สำเร็จ';
    case 'failed':
      return 'ไม่สำเร็จ';
    case 'error':
      return 'เกิดข้อผิดพลาด';
    case 'running':
      return 'กำลังทำงาน';
    case 'warning':
      return 'ควรตรวจสอบ';
    default:
      return status.trim().isEmpty ? 'ไม่ทราบสถานะ' : status;
  }
}

String formatSystemLogTimestamp(DateTime? timestamp) {
  if (timestamp == null) return 'ไม่พบเวลา';
  final local = timestamp.toLocal();
  String twoDigits(int value) => value.toString().padLeft(2, '0');
  return '${twoDigits(local.day)}/${twoDigits(local.month)}/${local.year} '
      '${twoDigits(local.hour)}:${twoDigits(local.minute)}:${twoDigits(local.second)}';
}
