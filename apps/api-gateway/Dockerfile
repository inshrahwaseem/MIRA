FROM node:22-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

WORKDIR /app

# Copy the entire workspace to ensure shared packages are available
COPY . .

# Install dependencies for the whole monorepo
RUN pnpm install --frozen-lockfile

# Build the api-gateway
RUN pnpm --filter api-gateway run build

# Set the working directory to the api-gateway for starting the app
WORKDIR /app/apps/api-gateway
EXPOSE 8080

CMD ["pnpm", "start"]
