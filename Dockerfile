FROM node:18-slim

# Set working directory
WORKDIR /usr/src/app

# Copy package files
COPY package*.json ./

# Install production dependencies
RUN npm install --omit=dev

# Copy project files
COPY . .

# Railway uses dynamic port
ENV PORT=3000

# Expose container port
EXPOSE 3000

# Start the server
CMD ["node", "server.js"]
