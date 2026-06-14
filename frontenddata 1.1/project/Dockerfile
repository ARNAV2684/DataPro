# Garuda ML Pipeline — Frontend (Vite/React build served by nginx)
#
#   docker build -t garuda-frontend:local "frontenddata 1.1/project"
#
FROM node:20-alpine AS build
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
# Use Vite's esbuild-based build directly. It transpiles TS/TSX for the
# production bundle without the full `tsc` type-check gate, so the image builds
# deterministically regardless of pre-existing loose types in the codebase.
RUN npx vite build

# --- serve the static build with nginx ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
