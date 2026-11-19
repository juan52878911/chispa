# Imagen del servidor de Chispa con L2 vía ONNX Runtime (-tags ort).
# El contexto es la raíz del repo (o lo que saca aws/empaquetar.sh): hacen falta server/, runtime/l1/, runtime/tok/,
# runtime/torre/ y format/chispa3/ por los `replace` de server/go.mod, y gomod-proxy/ con todos los módulos de
# server/go.sum (kindling y onnxruntime_go; GOPROXY no sale a internet).
#   docker build -f server/deploy/Dockerfile -t chispa-server \
#     --build-arg GO_DIGEST=sha256:... --build-arg ORT_NUPKG_SHA256=... .
# Hay que rellenar GO_DIGEST y DISTROLESS_DIGEST (índices de golang:${GO_VERSION}-bookworm y distroless/cc-debian12:nonroot) y ORT_NUPKG_SHA256 (sha256 del .nupkg Microsoft.ML.OnnxRuntime de NuGet; el repo de GitHub de ORT da 404 desde 2026-11-19): vacíos, el build falla.
ARG GO_VERSION=1.27
# onnxruntime_go lo fija server/go.mod (v1.38.0); ORT_VERSION debe ser compatible (igual que deploy/kindling/Dockerfile).
ARG ORT_VERSION=1.31.1
ARG GO_DIGEST=sha256:d9b6a1beb9474adc2a1ce842573639b44b2a0dd49b6669f51d3821baf9ec37c9
ARG DISTROLESS_DIGEST=sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f
ARG ORT_NUPKG_SHA256=6871ccbc97f3d5bdb4095a1dcec33aa78d56f6ed8de5980ef5fa1d9f2e7c679a

FROM golang:${GO_VERSION}-bookworm@${GO_DIGEST} AS build
ARG ORT_VERSION
ARG ORT_NUPKG_SHA256
WORKDIR /src
# Todos los módulos salen de la caché empaquetada y se verifican contra server/go.sum.
COPY gomod-proxy /gomod-proxy
ENV GOPROXY=file:///gomod-proxy \
    GOFLAGS=-mod=readonly \
    GOTOOLCHAIN=local \
    CGO_ENABLED=1
COPY format/chispa3 format/chispa3
COPY runtime/l1 runtime/l1
COPY runtime/tok runtime/tok
COPY runtime/torre runtime/torre
COPY server server
# ORT desde NuGet: el .nupkg es un zip con las .so en runtimes/linux-x64/native/ (onnxruntime_go trae su cabecera).
RUN [ -n "$ORT_NUPKG_SHA256" ] || { echo "falta ORT_NUPKG_SHA256: rellénelo (--build-arg)" >&2; exit 1; } \
 && apt-get update -qq && apt-get install -y -qq --no-install-recommends unzip \
 && curl -fsSL -o /tmp/ort.nupkg https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION} \
 && echo "$ORT_NUPKG_SHA256  /tmp/ort.nupkg" | sha256sum -c - \
 && mkdir -p /ort/lib && unzip -q -j /tmp/ort.nupkg runtimes/linux-x64/native/libonnxruntime.so -d /ort/lib
WORKDIR /src/server
RUN go build -tags ort -trimpath \
      -ldflags "-s -w -X github.com/juan52878911/chispa/server/decide.RutaORT=/usr/lib/libonnxruntime.so" \
      -o /out/chispa-server ./cmd/chispa-server \
 && go build -trimpath -ldflags "-s -w" -o /out/carga ./cmd/carga

# libstdc++ y libgcc que pide libonnxruntime.so, sin shell ni gestor de paquetes.
FROM gcr.io/distroless/cc-debian12:nonroot@${DISTROLESS_DIGEST}
ARG ORT_VERSION
COPY --from=build /ort/lib/libonnxruntime.so /usr/lib/libonnxruntime.so
COPY --from=build /out/chispa-server /out/carga /usr/local/bin/
# Modelo: monta un .chispa en /modelos/modelo.chispa (v3, o v2 solo con L0).
# Seguro por defecto: sin api_keys el servidor se niega a escuchar en 0.0.0.0 (FATAL sin api_keys). Dos caminos:
#   producción: monta una config y pásala:  docker run ... chispa-server -modelo /modelos/modelo.chispa -config /etc/chispa/config.json
#   pruebas:    añade -abierto (sin autenticación en /v1/decide; NUNCA en producción):  docker run ... chispa-server -abierto
# CMD lleva solo el modelo; los argumentos que pases al `docker run` lo sustituyen, así que repite -modelo si cambias CMD.
# Parada: SIGTERM drena en vuelo hasta -apagado (35 s, >= -timeout 30 s); la gracia de Docker debe ser >= apagado + 5 s:
#   docker run --stop-timeout 40 ...   (Kubernetes: terminationGracePeriodSeconds: 40)
STOPSIGNAL SIGTERM
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/chispa-server", "-addr", "0.0.0.0:8090", "-apagado", "35s"]
CMD ["-modelo", "/modelos/modelo.chispa"]
