# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all the project files to the working directory
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (8080 for Streamlit if you plan to use Streamlit)
EXPOSE 8080

# Command to run the application
CMD ["python", "chatbot_rag.py"]
