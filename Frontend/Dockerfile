# -----------------------------
# 1. Build Stage
# -----------------------------
FROM node:16-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy source code
COPY . .

# Build the React app
RUN npm run build

# -----------------------------
# 2. Production Stage
# -----------------------------
FROM nginx:stable-alpine

# Copy build output to Nginx HTML directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom Nginx config (optional but recommended)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
