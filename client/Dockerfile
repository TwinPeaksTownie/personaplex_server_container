FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Install OpenSSL for self-signed certs
RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

# Generate self-signed SSL certificate
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /app/key.pem -out /app/cert.pem \
    -subj "/C=US/ST=State/L=City/O=PersonaPlex/CN=localhost"

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
