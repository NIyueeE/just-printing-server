# Just Printing Server

[中文](README.中文.md) | English

A minimal, stateless printing relay service built with FastAPI.

## Features

- **Zero Persistence**: No database, no user system, no persistent storage
- **In-Memory Only**: All files live only in memory or temp directories, destroyed immediately after request
- **Environment Config**: All settings injected via environment variables
- **IPP Protocol**: Direct printing to IPP-compatible printers
- **Rate Limiting**: Built-in IP-based rate limiting
- **Podman Ready**: Easy deployment with Podman Compose

## Use Cases

### Wireless Home Printer
Connect your home printer directly to your network. Just-Printing-Server works with any IPP-compatible printer:
- **HP printers** (e.g., HP LaserJet, HP OfficeJet)
- **Brother printers** (e.g., Brother HL, MFC series)
- **Canon printers** (e.g., Canon PIXMA, imageRUNNER)
- **EPSON printers** (e.g., EPSON WorkForce, EcoTank)
- **Lenovo printers** (e.g., Lenovo LJ4000D, LJ2600D)

No need to install printer drivers or apps on your phone/laptop.

### Simple Deployment
Deploy on minimal hardware:
- Raspberry Pi (3B+/4/5)
- NAS devices (with Docker/Podman support)
- Mini PCs
- Any Linux server

### How It Works
1. Your printer is connected to WiFi and has an IPP address
2. Deploy Just-Printing-Server on a device on the same network
3. Access the web UI from any device (phone, tablet, laptop)
4. Upload photos or PDFs, adjust settings, and print - that's it!

## Project Structure

```
backend/
├── __init__.py       # Package initialization
├── main.py           # Application entry point
├── config.py         # Configuration management
├── models.py         # Pydantic data models
├── services.py       # Business logic services
└── routes.py         # API endpoints

frontend/
├── index.html        # Frontend main page
├── app.js            # Frontend JavaScript
└── styles.css        # Frontend styles
```

## Quick Start

### Option A: Run with Podman Compose

```bash
cp .env.example .env
# Edit .env with your printer settings

podman-compose up -d
```

### Option B: Run with uv (local development)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy environment file
cp .env.example .env
# Edit .env with your printer settings

# Initialize uv project (creates .venv)
uv sync

# Run the server
uv run uvicorn backend.main:app --host 0.0.0.0 --port 3001
```

### 3. Access the Web UI

Open `http://localhost:3001` in your browser.

The web interface provides:
- **Drag & drop file upload** - Support for JPG, PNG, and PDF files
- **PDF preview** - View merged documents before printing
- **Print settings** - Copies, sides, color, paper size, quality, orientation
- **Printer status** - Real-time printer online/offline status
- **Mobile responsive** - Works on phones and tablets

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth` | Authentication |
| POST | `/upload` | Upload files (images/PDFs) |
| GET | `/preview.pdf` | Preview merged PDF |
| GET | `/printer/status` | Check printer status |
| GET | `/printer/capabilities` | Get printer capabilities (supported print options) |
| POST | `/print` | Submit print job |
| POST | `/cancel` | Cancel session |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REGISTRY` | docker.io | Docker registry for base image (used in compose) |
| `FRONTEND_PATH` | frontend | Frontend static files path |
| `PRINTER_IPP_URL` | - | IPP URL of your printer |
| `PRINTER_NAME` | - | Printer display name |
| `ACCESS_TOKEN` | - | API access token |
| `MAX_UPLOAD_MB` | 50 | Maximum upload size (MB) |
| `RATE_LIMIT_PER_IP` | 5/minute | Rate limit per IP |
| `LOG_LEVEL` | INFO | Logging level |
| `IPP_DEFAULT_MEDIA` | iso-a4 | Default paper size |
| `IPP_DEFAULT_QUALITY` | normal | Default print quality |
| `IPP_DEFAULT_ORIENTATION` | portrait | Default print orientation |
| `IPP_USER_NAME` | fastapi | Username for print jobs |

## Printer Capabilities Detection

The `/printer/capabilities` endpoint automatically detects what printing options your printer supports. In the web UI:
- Supported options are displayed as clickable buttons
- Unsupported options appear grayed out and are disabled
- This allows users to know which settings are available on their specific printer

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Printing**: IPP protocol via `pyipp`
- **PDF Processing**: PyPDF2, img2pdf
- **Rate Limiting**: slowapi
- **Deployment**: Podman + Podman Compose

## License

MIT
