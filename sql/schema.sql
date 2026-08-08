CREATE DATABASE IF NOT EXISTS content_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE content_ai;

CREATE TABLE IF NOT EXISTS users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(100) NOT NULL UNIQUE,
  email VARCHAR(255) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL DEFAULT 'user',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX ix_users_user_id (user_id),
  INDEX ix_users_username (username),
  INDEX ix_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_contents (
  content_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  title VARCHAR(255) NOT NULL,
  video_url VARCHAR(1024) NULL,
  transcript TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_user_contents_content_id (content_id),
  CONSTRAINT fk_user_contents_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS keywords (
  keyword_id INT AUTO_INCREMENT PRIMARY KEY,
  keyword VARCHAR(100) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS content_keywords (
  id INT AUTO_INCREMENT PRIMARY KEY,
  content_id INT NOT NULL,
  keyword_id INT NOT NULL,
  score FLOAT NOT NULL DEFAULT 0,
  CONSTRAINT fk_content_keywords_content
    FOREIGN KEY (content_id) REFERENCES user_contents(content_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_content_keywords_keyword
    FOREIGN KEY (keyword_id) REFERENCES keywords(keyword_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS clusters (
  cluster_id INT AUTO_INCREMENT PRIMARY KEY,
  cluster_name VARCHAR(150) NOT NULL UNIQUE,
  description TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS dataset_contents (
  dataset_id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  video_url VARCHAR(1024) NULL,
  transcript TEXT NULL,
  category VARCHAR(100) NULL,
  source_platform VARCHAR(50) NOT NULL DEFAULT 'youtube',
  views INT NOT NULL DEFAULT 0,
  likes INT NOT NULL DEFAULT 0,
  comments INT NOT NULL DEFAULT 0,
  trend_score FLOAT NOT NULL DEFAULT 0,
  duration_seconds INT NULL,
  published_at DATETIME NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS trending_items (
  item_id INT AUTO_INCREMENT PRIMARY KEY,
  keyword VARCHAR(255) NOT NULL,
  score FLOAT NOT NULL DEFAULT 0,
  source VARCHAR(100) NOT NULL DEFAULT 'unknown',
  domain VARCHAR(100) NULL,
  fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  meta TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX ix_trending_items_keyword (keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS analysis_results (
  result_id INT AUTO_INCREMENT PRIMARY KEY,
  content_id INT NOT NULL,
  cluster_id INT NULL,
  dataset_id INT NULL,
  summary TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_analysis_results_content
    FOREIGN KEY (content_id) REFERENCES user_contents(content_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_analysis_results_cluster
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
    ON DELETE SET NULL,
  CONSTRAINT fk_analysis_results_dataset
    FOREIGN KEY (dataset_id) REFERENCES dataset_contents(dataset_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cluster_runs (
  run_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  algorithm VARCHAR(50) NOT NULL DEFAULT 'kmeans',
  n_clusters INT NOT NULL,
  feature_dimension INT NOT NULL DEFAULT 0,
  inertia FLOAT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_cluster_runs_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cluster_memberships (
  membership_id INT AUTO_INCREMENT PRIMARY KEY,
  run_id INT NOT NULL,
  cluster_id INT NOT NULL,
  content_id INT NULL,
  dataset_id INT NULL,
  item_text TEXT NOT NULL,
  top_terms TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_cluster_memberships_run
    FOREIGN KEY (run_id) REFERENCES cluster_runs(run_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_cluster_memberships_cluster
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_cluster_memberships_content
    FOREIGN KEY (content_id) REFERENCES user_contents(content_id)
    ON DELETE SET NULL,
  CONSTRAINT fk_cluster_memberships_dataset
    FOREIGN KEY (dataset_id) REFERENCES dataset_contents(dataset_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS recommendations (
  rec_id INT AUTO_INCREMENT PRIMARY KEY,
  content_id INT NOT NULL,
  recommended_keywords TEXT NULL,
  recommended_duration INT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_recommendations_content
    FOREIGN KEY (content_id) REFERENCES user_contents(content_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS system_configs (
  config_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  max_keywords INT NOT NULL DEFAULT 10,
  hook_duration INT NOT NULL DEFAULT 60,
  process_interval INT NOT NULL DEFAULT 24,
  notification_batch_size INT NOT NULL DEFAULT 50,
  youtube_region VARCHAR(2) NOT NULL DEFAULT 'TH',
  google_region VARCHAR(2) NOT NULL DEFAULT 'TH',
  tiktok_region VARCHAR(2) NOT NULL DEFAULT 'TH',
  enable_youtube_trending BOOLEAN NOT NULL DEFAULT TRUE,
  enable_google_trends BOOLEAN NOT NULL DEFAULT TRUE,
  enable_tiktok_trending BOOLEAN NOT NULL DEFAULT TRUE,
  auto_scan_interval_hours INT NOT NULL DEFAULT 6,
  asr_model_default VARCHAR(20) NOT NULL DEFAULT 'tiny',
  enable_model_toggle BOOLEAN NOT NULL DEFAULT TRUE,
  job_backend VARCHAR(20) NOT NULL DEFAULT 'inprocess',
  redis_url VARCHAR(255) NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_system_configs_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS system_logs (
  log_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  action VARCHAR(255) NOT NULL,
  status VARCHAR(50) NOT NULL,
  detail TEXT NULL,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_system_logs_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS notifications (
  notification_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  dataset_id INT NULL,
  title VARCHAR(255) NOT NULL,
  body TEXT NULL,
  link VARCHAR(255) NULL,
  type VARCHAR(50) NOT NULL DEFAULT 'system',
  payload TEXT NULL,
  message TEXT NOT NULL,
  topic VARCHAR(100) NOT NULL DEFAULT 'general',
  source_platform VARCHAR(50) NOT NULL DEFAULT 'system',
  trend_score FLOAT NOT NULL DEFAULT 0,
  is_read BOOLEAN NOT NULL DEFAULT FALSE,
  delivered_via_ws BOOLEAN NOT NULL DEFAULT FALSE,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_notifications_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_notifications_dataset
    FOREIGN KEY (dataset_id) REFERENCES dataset_contents(dataset_id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_trend_snapshots (
  snapshot_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  platform VARCHAR(50) NOT NULL,
  item_keys TEXT NOT NULL,
  snapshot_payload TEXT NOT NULL,
  last_checked_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_user_trend_snapshot_user_platform (user_id, platform),
  CONSTRAINT fk_user_trend_snapshots_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS followed_topics (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  match_type VARCHAR(20) NOT NULL,
  value VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX ix_followed_topics_value (value),
  CONSTRAINT fk_followed_topics_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
