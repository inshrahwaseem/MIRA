FROM node:20-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

WORKDIR /app
COPY . .

# Install dependencies for the whole monorepo
RUN pnpm install

# Build the api-gateway
RUN pnpm --filter api-gateway run build

WORKDIR /app/apps/api-gateway
EXPOSE 8080

CMD ["pnpm", "start"]
