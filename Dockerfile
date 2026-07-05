# -------------------------------------------------
# Base Image
# -------------------------------------------------
FROM python:3.12-slim

# -------------------------------------------------
# Working Directory
# -------------------------------------------------
WORKDIR /app

# -------------------------------------------------
# Copy requirements
# -------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------
# Install dependencies
# -------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Copy project files
# -------------------------------------------------
COPY . .

# -------------------------------------------------
# Expose FastAPI Port
# -------------------------------------------------
EXPOSE 8000

# -------------------------------------------------
# Start FastAPI
# -------------------------------------------------
# Notice we're using: python -m uvicorn instead of uvicorn.exe
# Because  Windows policy blocked uvicorn.exe. This approach is also portable.    
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]