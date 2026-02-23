# Just Printing Server

[中文](README.中文.md) | English

A minimal, stateless printing relay service built with FastAPI.

## Features

- **Zero Persistence**: No database, no user system, no persistent storage
- **In-Memory Only**: All files live only in memory or temp directories, destroyed immediately after request
- **Environment Config**: All settings injected via environment variables
- **IPP Protocol**: Direct printing to IPP-compatible printers
- **Rate Limiting**: Built-in IP-based rate limiting
- **Docker Ready**: Easy deployment with Docker Compose

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your printer settings
```

### 2. Run with Docker Compose

```bash
docker compose up -d
```

### 3. Access the Web UI

Open `http://localhost:8000` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth` | Authentication |
| POST | `/upload` | Upload files (images/PDFs) |
| GET | `/preview.pdf` | Preview merged PDF |
| GET | `/printer/status` | Check printer status |
| POST | `/print` | Submit print job |
| POST | `/cancel` | Cancel session |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRINTER_IPP_URL` | - | IPP URL of your printer |
| `PRINTER_NAME` | - | Printer display name |
| `ACCESS_TOKEN` | - | API access token |
| `MAX_UPLOAD_MB` | 50 | Maximum upload size (MB) |
| `RATE_LIMIT_PER_IP` | 5/minute | Rate limit per IP |
| `LOG_LEVEL` | INFO | Logging level |

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Printing**: IPP protocol via `pyipp`
- **PDF Processing**: PyPDF2, img2pdf
- **Rate Limiting**: slowapi
- **Deployment**: Docker + Docker Compose

## License

MIT
