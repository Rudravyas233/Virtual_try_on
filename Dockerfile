# Use lightweight Node image
FROM node:18-slim

# Set working directory
WORKDIR /usr/src/app

# Copy package files first (for Docker cache efficiency)
COPY package*.json ./

# Install dependencies
RUN npm install --omit=dev

# Copy all project files
COPY . .

# Railway provides PORT automatically
# Do NOT override it

# Expose generic port (Railway maps internally)
EXPOSE 8080

# Start server
CMD ["npm", "start"]
