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
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_contents (
  content_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  title VARCHAR(255) NOT NULL,
  video_url VARCHAR(255),
  transcript TEXT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_user_contents_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS keywords (
  keyword_id INT AUTO_INCREMENT PRIMARY KEY,
  keyword VARCHAR(100) NOT NULL UNIQUE
);

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
);

CREATE TABLE IF NOT EXISTS clusters (
  cluster_id INT AUTO_INCREMENT PRIMARY KEY,
  cluster_name VARCHAR(150) NOT NULL UNIQUE,
  description TEXT
);

CREATE TABLE IF NOT EXISTS dataset_contents (
  dataset_id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  video_url VARCHAR(255),
  transcript TEXT,
  category VARCHAR(100),
  source_platform VARCHAR(50) NOT NULL DEFAULT 'youtube',
  views INT NOT NULL DEFAULT 0,
  likes INT NOT NULL DEFAULT 0,
  comments INT NOT NULL DEFAULT 0,
  trend_score FLOAT NOT NULL DEFAULT 0,
  duration_seconds INT NULL,
  published_at DATETIME NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_results (
  result_id INT AUTO_INCREMENT PRIMARY KEY,
  content_id INT NOT NULL,
  cluster_id INT NULL,
  dataset_id INT NULL,
  summary TEXT,
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
);

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
);

CREATE TABLE IF NOT EXISTS cluster_memberships (
  membership_id INT AUTO_INCREMENT PRIMARY KEY,
  run_id INT NOT NULL,
  cluster_id INT NOT NULL,
  content_id INT NULL,
  dataset_id INT NULL,
  item_text TEXT NOT NULL,
  top_terms TEXT,
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
);

CREATE TABLE IF NOT EXISTS recommendations (
  rec_id INT AUTO_INCREMENT PRIMARY KEY,
  content_id INT NOT NULL,
  recommended_keywords TEXT,
  recommended_duration INT,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_recommendations_content
    FOREIGN KEY (content_id) REFERENCES user_contents(content_id)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS system_configs (
  config_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  max_keywords INT NOT NULL DEFAULT 10,
  hook_duration INT NOT NULL DEFAULT 60,
  process_interval INT NOT NULL DEFAULT 24,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_system_configs_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS system_logs (
  log_id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  action VARCHAR(255) NOT NULL,
  status VARCHAR(50) NOT NULL,
  detail TEXT,
  timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_system_logs_user
    FOREIGN KEY (user_id) REFERENCES users(user_id)
    ON DELETE SET NULL
);
