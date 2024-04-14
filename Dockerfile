FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose port 80 (or the port your FastAPI app is running on)
EXPOSE 80

# Command to run your FastAPI application
CMD ["uvicorn", "main:app",  "--port", "80"]