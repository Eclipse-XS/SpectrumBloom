FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

USER root
RUN rm -rf /usr/share/nginx/html/*
COPY --chown=101:101 nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=101:101 --from=build /app/dist/frontend/browser /usr/share/nginx/html
USER 101

EXPOSE 8501

CMD ["nginx", "-g", "daemon off;"]
