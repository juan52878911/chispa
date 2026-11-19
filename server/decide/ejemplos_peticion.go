package decide

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"sync"
)

// Ejemplos en la petición (`examples[]` por pregunta): el cliente trae los ejemplos de oro en cada llamada en vez de un banco fuera de línea.
//
// Regla si llegan los dos (banco de la tarea y examples): los de la petición SUSTITUYEN al banco de esa pregunta, no se mezclan. Con examples la pregunta
// no necesita estar en `knn.tareas`; si lo está se usan su umbral u y su marca «siempre», pero no sus umbrales de score (son del banco) ni su fondo.
// Sin examples todo sigue como antes.
//
// Los vectores de los ejemplos se guardan en una caché LRU con tope de memoria (knn.cache_vectores_mb) con clave sha256 del texto ya preparado para el
// codificador + codificador + revisión: la misma lista de ejemplos en la petición siguiente no vuelve a codificarse (con Qwen en CPU son ≈ 1 s por ejemplo).

// EjemploPeticion es un elemento de `examples[]`: el texto y la opción (id) que le corresponde.
type EjemploPeticion struct {
	Texto    string
	Etiqueta string
}

// E5Revision es la revisión de multilingual-e5-small de los bancos y de la caché de vectores.
const E5Revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

// CacheVectoresMBPorDefecto es el tope de la caché de vectores de ejemplos (knn.cache_vectores_mb = 0).
const CacheVectoresMBPorDefecto = 64

// hashEjemplos identifica una lista de ejemplos (para la caché de respuestas por nonce).
func hashEjemplos(ejs []EjemploPeticion) string {
	h := sha256.New()
	for _, e := range ejs {
		h.Write([]byte(strconv.Itoa(len(e.Texto)) + ":" + e.Texto + strconv.Itoa(len(e.Etiqueta)) + ":" + e.Etiqueta))
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// cacheVectores: vectores por clave de contenido con desalojo LRU por bytes y deduplicación de codificaciones simultáneas de la misma clave.
type cacheVectores struct {
	lru *LRU[[]float32]
	mu  sync.Mutex
	en  map[string]*vueloVector
}

type vueloVector struct {
	done chan struct{}
	v    []float32
	err  error
}

func nuevaCacheVectores(mb int) *cacheVectores {
	if mb <= 0 {
		mb = CacheVectoresMBPorDefecto
	}
	return &cacheVectores{lru: NewLRUBytes(mb<<20, func(v []float32) int { return 4 * len(v) }), en: map[string]*vueloVector{}}
}

// da devuelve el vector de la clave: de la caché, del vuelo en curso de otra petición o de f.
func (c *cacheVectores) da(ctx context.Context, clave string, f func() ([]float32, error)) ([]float32, error) {
	if v, ok := c.lru.Get(clave); ok {
		return v, nil
	}
	c.mu.Lock()
	if v, ok := c.lru.Get(clave); ok {
		c.mu.Unlock()
		return v, nil
	}
	if w, ok := c.en[clave]; ok {
		c.mu.Unlock()
		select {
		case <-w.done:
			return w.v, w.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	w := &vueloVector{done: make(chan struct{})}
	c.en[clave] = w
	c.mu.Unlock()
	w.v, w.err = f()
	if w.err == nil {
		c.lru.Put(clave, w.v)
	}
	c.mu.Lock()
	delete(c.en, clave)
	c.mu.Unlock()
	close(w.done)
	return w.v, w.err
}

// revisionCodificador: la revisión del modelo que entra en la clave de la caché.
func (k *L2KNN) revisionCodificador() string {
	if k.cfg.NombreCodificador() == CodificadorQwen3Emb {
		return QwenRevision
	}
	return E5Revision
}

// textoPreparado es lo que recibe el codificador para un texto: con Qwen la consulta con instrucción y pregunta; con e5 el texto con las mayúsculas normalizadas.
func (k *L2KNN) textoPreparado(pregunta, texto string) string {
	if _, ok := k.cod.(CodificadorConsulta); ok {
		return QwenConsulta(pregunta, texto)
	}
	return NormalizarMayusculas(texto)
}

func (k *L2KNN) claveVector(pregunta, texto string) string {
	h := sha256.Sum256([]byte(k.textoPreparado(pregunta, texto)))
	return hex.EncodeToString(h[:]) + "|" + k.cfg.NombreCodificador() + "|" + k.revisionCodificador()
}

// codificaEjemplo da el vector de un ejemplo con la misma ruta que la consulta, por la caché.
func (k *L2KNN) codificaEjemplo(ctx context.Context, pregunta, texto string) ([]float32, error) {
	return k.cacheVec.da(ctx, k.claveVector(pregunta, texto), func() ([]float32, error) {
		if cq, ok := k.cod.(CodificadorConsulta); ok {
			return cq.CodificarConsulta(ctx, pregunta, texto)
		}
		return k.cod.Codificar(ctx, texto)
	})
}

// bancoDePeticion monta el banco (todo oro, en el orden de la petición) con los vectores de los ejemplos.
func (k *L2KNN) bancoDePeticion(ctx context.Context, p Pregunta) (*BancoKNN, error) {
	var b *BancoKNN
	for i, e := range p.Ejemplos {
		v, err := k.codificaEjemplo(ctx, p.Texto, e.Texto)
		if err != nil {
			return nil, fmt.Errorf("ejemplo %d: %w", i, err)
		}
		if b == nil {
			b = &BancoKNN{Dim: len(v), X: make([]float32, 0, len(v)*len(p.Ejemplos))}
		} else if len(v) != b.Dim {
			return nil, errDimKNN
		}
		b.X = append(b.X, v...)
		b.Etiq = append(b.Etiq, e.Etiqueta)
	}
	if b != nil {
		b.NOro = len(b.Etiq)
	}
	return b, nil
}

// ejemplosDePeticion son todos los examples de la petición como ejemplos fs, en su orden.
func ejemplosDePeticion(p Pregunta) []EjemploFS {
	ops := make([]string, len(p.Opciones))
	for i, o := range p.Opciones {
		ops[i] = o.ID
	}
	out := make([]EjemploFS, len(p.Ejemplos))
	for i, e := range p.Ejemplos {
		out[i] = EjemploFS{ID: "peticion-" + strconv.Itoa(i+1), Pregunta: p.Texto, Tipo: p.Kind, Texto: e.Texto, Opciones: ops, Etiqueta: e.Etiqueta}
	}
	return out
}

// DemosDePeticion elige hasta k ejemplos de la petición para el few-shot FIJO de L3 (sin kNN: modo cpu no lo usa, y en modo gpu solo con l3.demos_fijas o sin codificador):
// una vuelta por clase (en el orden en que aparecen) tomando el primer ejemplo aún no usado de cada una, hasta k; sin azar, así que la misma petición da siempre el mismo prompt.
// Con k ≤ 0 no hay demostraciones.
func DemosDePeticion(p Pregunta, k int) []EjemploFS {
	if k <= 0 || len(p.Ejemplos) == 0 {
		return nil
	}
	todos := ejemplosDePeticion(p)
	var clases []string
	por := map[string][]int{}
	for i, e := range p.Ejemplos {
		if _, ok := por[e.Etiqueta]; !ok {
			clases = append(clases, e.Etiqueta)
		}
		por[e.Etiqueta] = append(por[e.Etiqueta], i)
	}
	var out []EjemploFS
	for ronda := 0; len(out) < k; ronda++ {
		hubo := false
		for _, c := range clases {
			if ronda < len(por[c]) && len(out) < k {
				e := todos[por[c][ronda]]
				e.ID = "peticion-" + strconv.Itoa(len(out)+1)
				out = append(out, e)
				hubo = true
			}
		}
		if !hubo {
			break
		}
	}
	return out
}
