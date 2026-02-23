# Just Printing Server

中文 | [English](README.md)

一个极简、无状态的打印中继服务，基于 FastAPI 构建。

## 特性

- **零持久化**: 无数据库、无用户体系、无持久化存储
- **纯内存处理**: 所有文件仅存于内存或临时目录，请求结束后立即销毁
- **环境变量配置**: 所有配置通过环境变量注入
- **IPP 协议**: 直接连接支持 IPP 协议的打印机
- **限流保护**: 内置基于 IP 的速率限制
- **Docker 就绪**: 支持 Docker Compose 一键部署

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 配置你的打印机信息
```

### 2. 使用 Docker Compose 启动

```bash
docker compose up -d
```

### 3. 访问 Web 界面

在浏览器中打开 `http://localhost:8000`。

## API 接口

| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/auth` | 认证 |
| POST | `/upload` | 上传文件（图片/PDF） |
| GET | `/preview.pdf` | 预览合并后的 PDF |
| GET | `/printer/status` | 查询打印机状态 |
| POST | `/print` | 提交打印任务 |
| POST | `/cancel` | 取消会话 |

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PRINTER_IPP_URL` | - | 打印机的 IPP 地址 |
| `PRINTER_NAME` | - | 打印机显示名称 |
| `ACCESS_TOKEN` | - | API 访问令牌 |
| `MAX_UPLOAD_MB` | 50 | 最大上传大小（MB） |
| `RATE_LIMIT_PER_IP` | 5/minute | 每 IP 限流频率 |
| `LOG_LEVEL` | INFO | 日志级别 |

## 技术栈

- **后端**: FastAPI + Python 3.11
- **打印**: IPP 协议（`pyipp`）
- **PDF 处理**: PyPDF2, img2pdf
- **限流**: slowapi
- **部署**: Docker + Docker Compose

## 许可证

MIT
