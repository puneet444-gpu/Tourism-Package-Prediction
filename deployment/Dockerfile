
# ==============================
# Base image (lightweight Python 3.9)
# ==============================
FROM python:3.9-slim

# ==============================
# Set initial working directory
# ==============================
WORKDIR /app

# ==============================
# Copy and install dependencies
# ==============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# Copy all application files
# ==============================
COPY . .

# ==============================
# Create non-root user for security
# ==============================
RUN useradd -m -u 1000 user
USER user

# ==============================
# Set environment variables
# ==============================
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Ensure user has correct working directory
WORKDIR $HOME/app

# ==============================
# Expose port for Streamlit
# ==============================
EXPOSE 8501

# ==============================
# Run the Streamlit app
# ==============================
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
