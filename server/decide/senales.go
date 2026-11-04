package decide

import "chispa/runtime/tok"

// Señales del texto en la respuesta: si alguna capa que corrió truncó el texto y qué fracción de sus tokens era <unk>.

// SenalTexto es lo que una capa sabe del texto que tokenizó: tokens sin truncar, cuántos <unk> y si lo truncó (vio solo una parte).
type SenalTexto struct {
	Tokens, Unk int
	Truncado    bool
}

// ConSenales lo implementan las capas que tokenizan el texto (L1 y L2). ok=false si la capa no tokeniza (stub, sin tokenizador).
type ConSenales interface {
	Senales(texto string) (s SenalTexto, ok bool)
}

// TokenizadorSenales lo implementa el tokenizador de L2 que sabe contar (TokenizadorE5).
type TokenizadorSenales interface {
	Senales(texto string) SenalTexto
}

// Senales de TokenizadorE5: la consulta («query: »+texto) frente a su max_len.
func (e TokenizadorE5) Senales(texto string) SenalTexto {
	n, unk, trunc := e.t.Senales(tok.Prefijo, texto, maxOPorDefecto(e.consulta))
	return SenalTexto{Tokens: n, Unk: unk, Truncado: trunc}
}

// Senales de L1: con «trozos» el texto se lee entero (no se trunca); con «cabeza» o «cabeza_cola» sí se pierde una parte.
func (l *L1) Senales(texto string) (SenalTexto, bool) {
	if l == nil || l.m == nil {
		return SenalTexto{}, false
	}
	n, unk, trunc := l.m.Senales(texto)
	return SenalTexto{Tokens: n, Unk: unk, Truncado: trunc && !leeEntero(l.largo)}, true
}

// Senales de L2: las del tokenizador de la consulta, si sabe contar.
func (l *L2) Senales(texto string) (SenalTexto, bool) {
	if l == nil || l.tok == nil {
		return SenalTexto{}, false
	}
	ts, ok := l.tok.(TokenizadorSenales)
	if !ok {
		return SenalTexto{}, false
	}
	s := ts.Senales(texto)
	s.Truncado = s.Truncado && !leeEntero(l.largo)
	return s, true
}

func leeEntero(tl TextoLargo) bool { return tl.Estrategia == tok.EstTrozos }

// senalesRespuesta acumula las señales de las capas que corrieron: truncado si alguna truncó, tokens el máximo, unk_pct el mayor.
type senalesRespuesta struct {
	hay      bool
	truncado bool
	tokens   int
	unkPct   float64
}

func (a *senalesRespuesta) suma(s SenalTexto) {
	a.hay = true
	a.truncado = a.truncado || s.Truncado
	a.tokens = max(a.tokens, s.Tokens)
	if s.Tokens > 0 {
		a.unkPct = max(a.unkPct, float64(s.Unk)/float64(s.Tokens))
	}
}

func (a senalesRespuesta) aplica(r *Resultado) {
	if !a.hay {
		return
	}
	u := a.unkPct
	r.UnkPct = &u
	if a.truncado {
		r.Truncado = true
		r.TokensOriginales = a.tokens
	}
}
