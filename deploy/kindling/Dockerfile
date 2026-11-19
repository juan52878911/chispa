# Imagen de chispa-server para microVM de kindling (rootfs OCI). Contexto = raíz del repo:
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/kindling/Dockerfile --target imagen-cpu \
#     -t chispa-server:cpu .      # sin ONNX Runtime: binario estático (CGO_ENABLED=0)
#   ... --target imagen-ort -t chispa-server:ort .   # con -tags ort: enlaza libonnxruntime.so (glibc)
# Hacen falta format/chispa3, runtime/l1, runtime/tok, runtime/torre y server por los `replace` de server/go.mod, y
# gomod-proxy/ con TODOS los módulos de server/go.sum (kindling y onnxruntime_go): GOPROXY no sale a internet.
# No construida aquí (sin red). Cadena de suministro: hay que rellenar (--build-arg) antes de construir, o el build falla:
#   GO_DIGEST=sha256:...  DEBIAN_DIGEST=sha256:...  (digest del índice multi-arch de golang:${GO_VERSION}-bookworm
#     y debian:bookworm-slim: `docker buildx imagetools inspect <imagen>`)
#   ORT_NUPKG_SHA256 (sha256 del .nupkg Microsoft.ML.OnnxRuntime de NuGet, trae x64 y arm64; imagen-ort; el repo de GitHub de ORT da 404 desde 2026-11-19)
ARG GO_VERSION=1.27
ARG ORT_VERSION=1.31.1
# Sin BuildKit (docker build clásico, sin buildx) las variables automáticas llegan vacías: estos valores solo cubren ese caso;
# BuildKit/buildx los sustituye por los suyos.
ARG BUILDPLATFORM=linux/amd64
ARG GO_DIGEST=sha256:d9b6a1beb9474adc2a1ce842573639b44b2a0dd49b6669f51d3821baf9ec37c9
ARG DEBIAN_DIGEST=sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3
ARG ORT_NUPKG_SHA256=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-bookworm@${GO_DIGEST} AS cpu
ARG TARGETOS TARGETARCH
WORKDIR /src
COPY gomod-proxy /gomod-proxy
ENV GOPROXY=file:///gomod-proxy GOFLAGS=-mod=readonly GOTOOLCHAIN=local CGO_ENABLED=0
COPY format/chispa3 format/chispa3
COPY runtime/l1 runtime/l1
COPY runtime/tok runtime/tok
COPY runtime/torre runtime/torre
COPY server server
COPY deploy/kindling/sondeo deploy/kindling/sondeo
RUN cd server && GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -ldflags "-s -w" -o /out/chispa-server ./cmd/chispa-server
RUN cd deploy/kindling/sondeo && GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath -ldflags "-s -w" -o /out/sondeo .

# ORT=1: cgo exige compilar en la plataforma destino (buildx con qemu o un builder nativo arm64).
FROM golang:${GO_VERSION}-bookworm@${GO_DIGEST} AS ort
ARG TARGETARCH ORT_VERSION ORT_NUPKG_SHA256
WORKDIR /src
COPY gomod-proxy /gomod-proxy
ENV GOPROXY=file:///gomod-proxy GOFLAGS=-mod=readonly GOTOOLCHAIN=local CGO_ENABLED=1
COPY format/chispa3 format/chispa3
COPY runtime/l1 runtime/l1
COPY runtime/tok runtime/tok
COPY runtime/torre runtime/torre
COPY server server
COPY deploy/kindling/sondeo deploy/kindling/sondeo
# ORT desde NuGet (Microsoft.ML.OnnxRuntime): un zip con runtimes/linux-x64/native/ y runtimes/linux-arm64/native/.
RUN case "$TARGETARCH" in amd64) A=x64;; arm64) A=arm64;; *) exit 1;; esac \
 && [ -n "$ORT_NUPKG_SHA256" ] || { echo "falta ORT_NUPKG_SHA256: rellénelo (--build-arg)" >&2; exit 1; } \
 && apt-get update -qq && apt-get install -y -qq --no-install-recommends unzip \
 && curl -fsSL -o /tmp/ort.nupkg https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION} \
 && echo "$ORT_NUPKG_SHA256  /tmp/ort.nupkg" | sha256sum -c - \
 && mkdir -p /ort/lib && unzip -q -j /tmp/ort.nupkg runtimes/linux-${A}/native/libonnxruntime.so -d /ort/lib
# onnxruntime_go lo fija server/go.mod (v1.38.0); no se hace `go get` al construir.
RUN cd server && go build -tags ort -trimpath -ldflags "-s -w -X github.com/juan52878911/chispa/server/decide.RutaORT=/usr/lib/libonnxruntime.so" \
      -o /out/chispa-server ./cmd/chispa-server
RUN cd deploy/kindling/sondeo && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/sondeo .
# libstdc++ y libgcc_s (las pide libonnxruntime.so) salen del builder: su ruta depende de la arquitectura.
RUN mkdir -p /ortlib && cp /ort/lib/libonnxruntime.so /ortlib/libonnxruntime.so \
 && cp /usr/lib/*-linux-gnu/libstdc++.so.6 /usr/lib/*-linux-gnu/libgcc_s.so.1 /ortlib/

# Base con /bin/sh (entrada.sh lee las variables de entorno); usuario sin privilegios 10001.
FROM debian:bookworm-slim@${DEBIAN_DIGEST} AS base
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin chispa \
 && mkdir /modelos && chmod 755 /modelos
COPY deploy/kindling/entrada.sh /usr/local/bin/entrada.sh
ENV CHISPA_PUERTO=8090 CHISPA_MODELO=/modelos/modelo.chispa CHISPA_TOK=/modelos/e5.chispa-tok GOMEMLIMIT=400MiB
USER 10001:10001
VOLUME /modelos
EXPOSE 8090
STOPSIGNAL SIGTERM
# Drenaje: entrada.sh pasa -apagado 35s (>= -timeout 30s); da a Docker >= 40 s (--stop-timeout 40; Kubernetes terminationGracePeriodSeconds: 40).
ENTRYPOINT ["/usr/local/bin/entrada.sh"]

FROM base AS imagen-cpu
COPY --from=cpu /out/chispa-server /out/sondeo /usr/local/bin/
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 CMD ["/usr/local/bin/sondeo"]

FROM base AS imagen-ort
USER root
COPY --from=ort /ortlib/ /usr/lib/chispa/
COPY --from=ort /out/chispa-server /out/sondeo /usr/local/bin/
# libonnxruntime.so debe estar en /usr/lib (RutaORT); las dos libs de C++ van junto a ella.
RUN mv /usr/lib/chispa/libonnxruntime.so /usr/lib/chispa/libstdc++.so.6 /usr/lib/chispa/libgcc_s.so.1 /usr/lib/ \
 && rmdir /usr/lib/chispa && ldconfig
# Misma ruta que el -ldflags; chispa-server la lee (CHISPA_ORT_LIB o -ort-lib) para L2 y la torre. Se puede cambiar al ejecutar.
ENV CHISPA_ORT_LIB=/usr/lib/libonnxruntime.so
USER 10001:10001
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 CMD ["/usr/local/bin/sondeo"]
