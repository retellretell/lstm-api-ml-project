# Environment Variables for Financial AI Prediction System
# Copy this file to .env and fill in your actual values

# API Keys
DART_API_KEY=your_dart_api_key_here
BOK_API_KEY=your_bank_of_korea_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# PostgreSQL (optional)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=financial_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Application Settings
APP_ENV=production
DEBUG=false
SECRET_KEY=your_secret_key_here
LOG_LEVEL=INFO

# Model Configuration
MODEL_CHECKPOINT_DIR=./models/checkpoints
MODEL_CACHE_DIR=./models/cache
MAX_MODEL_VERSIONS=10

# Data Settings
DATA_UPDATE_INTERVAL=300  # seconds
CACHE_TTL=3600  # seconds
MAX_CACHE_SIZE=1000  # MB

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
TF_GPU_MEMORY_LIMIT=4096  # MB

# API Rate Limits
DART_RATE_LIMIT_PER_SECOND=10
DART_RATE_LIMIT_PER_DAY=10000
BOK_RATE_LIMIT_PER_SECOND=5
BOK_RATE_LIMIT_PER_DAY=5000

# Notification Settings (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=admin@example.com

# Monitoring
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
ENABLE_GRAFANA=false
GRAFANA_PORT=3000

# Feature Flags
ENABLE_REAL_TIME_DATA=true
ENABLE_GPU_ACCELERATION=true
ENABLE_MIXED_PRECISION=true
ENABLE_REDIS_CACHE=true
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_BACKTESTING=true
ENABLE_PAPER_TRADING=false

# Security
ENABLE_HTTPS=false
SSL_CERT_PATH=
SSL_KEY_PATH=
ALLOWED_ORIGINS=http://localhost:8501,http://localhost:3000
SESSION_TIMEOUT=3600  # seconds

# External Services
SLACK_WEBHOOK_URL=
TELEGRAM_BOT_TOKEN=
DISCORD_WEBHOOK_URL=

# Data Sources Priority
DATA_SOURCE_PRIORITY=pykrx,yfinance,alpha_vantage,manual

# Timezone
TZ=Asia/Seoul