# AI Data Modeler — dashboard image.
#
# Next.js 16 / React 19. The dashboard is a pure API client, so the only wiring it
# needs is the API origin. That value is inlined into the client bundle at *build*
# time, which is why it is an ARG here and not only an environment variable: an
# image built against localhost cannot be repointed at a different API by
# restarting the container.
#
#   docker build --build-arg NEXT_PUBLIC_API_URL=https://api.example.com -t automl-frontend frontend/

# ---------------------------------------------------------------------------
# Stage 1 — dependencies
# ---------------------------------------------------------------------------
FROM node:22-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json ./
# `npm ci` for a lockfile-exact install; a dashboard that builds differently on
# two machines is not worth debugging.
RUN npm ci

# ---------------------------------------------------------------------------
# Stage 2 — build
# ---------------------------------------------------------------------------
FROM node:22-alpine AS build

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# `next start` reads ./public unconditionally; create it so the copy below works
# whether or not the project has static assets yet.
RUN mkdir -p public && npm run build

# ---------------------------------------------------------------------------
# Stage 3 — runtime
# ---------------------------------------------------------------------------
FROM node:22-alpine AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 --ingroup nodejs nextjs

WORKDIR /app

# Production dependencies only: the build toolchain and dev types stay behind.
COPY --from=build /app/package.json /app/package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

COPY --from=build --chown=nextjs:nodejs /app/.next ./.next
COPY --from=build --chown=nextjs:nodejs /app/public ./public
COPY --from=build /app/next.config.mjs ./next.config.mjs

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -q -O /dev/null "http://127.0.0.1:${PORT}/" || exit 1

CMD ["npm", "run", "start"]
