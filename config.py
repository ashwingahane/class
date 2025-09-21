# config.py - Configuration Settings
"""
Configuration settings for the Smart Attendance System
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'smart_attendance_system_2024'
    DATABASE_NAME = 'enhanced_attendance.db'
    
    # Face Recognition Settings
    SAMPLES_PER_STUDENT = 8
    QUALITY_THRESHOLD = 0.6
    CONFIDENCE_THRESHOLD = 0.75
    
    # Attendance Session Settings
    DEFAULT_SESSION_DURATION = 10  # minutes
    DUPLICATE_PREVENTION_HOURS = 24  # hours
    SCHEDULER_CHECK_INTERVAL = 30  # seconds
    
    # Available Subjects
    SUBJECTS = ['Math', 'Python', 'Java']
    
    # Camera Settings
    CAMERA_INDEX = 0
    FACE_MIN_SIZE = (80, 80)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # File Paths
    FACE_SAMPLES_DIR = 'face_samples'
    SESSION_LOGS_DIR = 'session_logs'
    TEMPLATES_DIR = 'templates'
    STATIC_DIR = 'static'
    
    # Teacher Authentication
    TEACHER_EMAIL = 'admin@123.com'
    TEACHER_PASSWORD = '123'
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    
    # Database Settings
    DB_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Scheduler Settings
    AUTO_START_SCHEDULER = True
    UPCOMING_CLASS_MINUTES = 15  # Check for classes starting within 15 minutes
    
    # System Settings
    DEBUG_MODE = True
    LOG_LEVEL = 'INFO'
    HOST = '0.0.0.0'
    PORT = 5000

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production_secret_key_change_this'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_NAME = 'test_attendance.db'

# Default configuration
config = Config()