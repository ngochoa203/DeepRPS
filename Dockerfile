# Multi-stage build for full-stack app
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY FE/package*.json ./
RUN npm ci --only=production

COPY FE/ ./
RUN npm run build

# Python backend
FROM python:3.9-slim

WORKDIR /app

# Install Python dependencies
COPY AI/gamebrain/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python code
COPY AI/gamebrain/ ./gamebrain/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./static

# Create data directory for persistence
RUN mkdir -p /app/data

# Environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "gamebrain.server.main:app", "--host", "0.0.0.0", "--port", "8000"]