FROM node:20-alpine AS deps
# Check https://github.com/nodejs/docker-node/tree/b4117f9333da4138b03a546ec926ef50a31506c3#nodealpine to understand why libc6-compat might be needed.
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# We use npm run build, assuming an Express/Node app or similar build process
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 api-gateway

# Set the correct permission for prerender cache
RUN chown api-gateway:nodejs /app

COPY --from=builder --chown=api-gateway:nodejs /app/dist ./dist
COPY --from=builder --chown=api-gateway:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=api-gateway:nodejs /app/package.json ./package.json

USER api-gateway

EXPOSE 3001
ENV PORT 3001
ENV HOSTNAME "0.0.0.0"

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:3001/health || exit 1

CMD ["node", "dist/index.js"]
