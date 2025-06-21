# AI Video Translation Service

A sophisticated FastAPI-based application that automatically translates videos from one language to another using state-of-the-art AI models. Upload an MP4 video and receive a fully dubbed version in your target language with preserved timing and speaker characteristics.

## 🎯 Features

- **Automatic Video Translation**: Upload MP4 videos and get dubbed versions in 10+ languages
- **Speaker Preservation**: Maintains speaker identity and gender characteristics
- **Real-time Progress**: Live progress updates via WebSocket during translation
- **Web Interface**: Modern drag-and-drop frontend with progress tracking
- **Job Management**: Persistent job storage with status monitoring and download
- **Multi-Language Support**: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg
- Docker (optional)
- Hugging Face account and token

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-video-translation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your Hugging Face token
   export HUGGING_FACE_TOKEN="your_token_here"
   ```

4. **Run the application**
   ```bash
   python -m app.main
   ```

5. **Access the web interface**
   ```
   http://localhost:8000
   ```

### Docker Deployment

```bash
# Set your Hugging Face token
export HUGGING_FACE_TOKEN="your_token_here"

# Run with Docker Compose
docker-compose up
```

## 🎬 How It Works

1. **Upload**: Drag and drop an MP4 video file
2. **Configure**: Select source and target languages, AI models
3. **Process**: 7-stage AI pipeline processes your video:
   - Audio/video separation and speaker diarization
   - Speech-to-text transcription with gender classification
   - Text translation using NLLB models
   - Voice assignment based on speaker characteristics
   - Text-to-speech generation with MMS models
   - Audio/video recombination and optimization
4. **Download**: Get your translated video with preserved timing

## 🔧 Technology Stack

- **Backend**: FastAPI, Python 3.8+, SQLite, WebSockets
- **AI Models**: 
  - Whisper (Speech-to-Text)
  - NLLB (Translation)
  - MMS (Text-to-Speech)
  - PyAnnote (Speaker Diarization)
- **Processing**: FFmpeg, MoviePy, PyDub
- **Frontend**: HTML5, CSS3, JavaScript
- **Infrastructure**: Docker, Docker Compose

## 📊 Performance

- **Small videos** (< 5 min): 2-5 minutes processing
- **Medium videos** (5-15 min): 5-15 minutes processing
- **Resource efficient**: CPU-optimized, no GPU required
- **Concurrent processing**: Configurable job queue limits

## 🔗 API Endpoints

- `POST /api/v1/upload` - Upload video for translation
- `GET /api/v1/jobs/{job_id}/status` - Get job status
- `GET /api/v1/jobs/{job_id}/download` - Download translated video
- `WS /api/v1/jobs/{job_id}/progress` - Real-time progress updates
- `GET /health` - Health check endpoint

## 📁 Project Structure

```
app/
├── main.py                 # FastAPI application entry point
├── api/routes/            # API endpoint definitions
├── models/                # Data models and schemas
├── services/              # Core business logic
│   ├── processing/        # Audio/video processing pipeline
│   ├── stt/              # Speech-to-text services
│   ├── translation/      # Translation services
│   └── tts/              # Text-to-speech services
├── static/               # Frontend assets
└── templates/            # HTML templates
```

## 🔧 Configuration

Key environment variables:

```bash
# Required
HUGGING_FACE_TOKEN=your_token_here

# Optional
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
OUTPUT_DIRECTORY=output/
MAX_FILE_SIZE_MB=200
DEVICE=cpu
CLEAN_INTERMEDIATE_FILES=false
```

## 📖 Documentation

Comprehensive documentation is available in the `codebase_explained/` folder:

- [Project Overview](codebase_explained/overview.md) - Detailed project description and technologies
- [Component Breakdown](codebase_explained/components.md) - System components and architecture
- [Entry Point & Initialization](codebase_explained/entry_point.md) - Startup process and configuration
- [Execution Flow](codebase_explained/execution_flow.md) - Complete translation workflow
- [Architecture & Design Patterns](codebase_explained/architecture.md) - Design patterns and scalability
- [Flow Diagrams](codebase_explained/flow_diagrams.md) - Visual system flows and processes

## 🐳 Production Deployment

For production deployment, consider:

- Use Gunicorn with multiple workers
- Set up reverse proxy (Nginx)
- Configure proper logging and monitoring
- Use external database (PostgreSQL)
- Set up file storage (S3, GCS)
- Configure auto-scaling based on queue length

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the [documentation](codebase_explained/) for detailed information
- Review the health endpoint: `GET /health`
- Check logs for error details
- Ensure FFmpeg is properly installed
- Verify Hugging Face token is valid

## 🎯 Roadmap

- [ ] Support for additional video formats
- [ ] Batch processing capabilities
- [ ] Advanced voice cloning options
- [ ] Multi-speaker voice assignment
- [ ] Integration with cloud storage providers
- [ ] Performance optimizations and GPU support

---

**Built with ❤️ using FastAPI and state-of-the-art AI models**
