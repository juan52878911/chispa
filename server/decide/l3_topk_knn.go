package decide

import "context"

// Preselección top-k por kNN (R10-T3b, V4): en una tarea choice con más de L3TopKKNNDesde opciones, L3 solo ve las top_k_knn mejores del ranking del kNN,
// como `lineas_base_v2_pred.cmd_topk` (k = 10): probabilidad de voto descendente y, a igualdad, similitud al centroide de oro de la clase; a igualdad de
// ambas, el orden de la fila. Las opciones elegidas se muestran en el orden de la pregunta (Python las ordena alfabéticamente; en la batería v2 las filas de
// finanzas ya lo están). La etiqueta final es una de ellas; las demás salen con p = 0.

// ConUmbrales activa el score calibrado de L3: f da, para el texto de una pregunta, los umbrales de su tarea (nil = sin calibrar). Solo afecta al modo logprobs.
func (l *L3LLM) ConUmbrales(f func(pregunta string) *UmbralesScore) *L3LLM {
	l.umbralesDe = f
	return l
}

// topKKNN es el k efectivo de la preselección por kNN (0 = apagada).
func (l *L3LLM) topKKNN() int {
	if l.cfg.TopKKNN == nil {
		return 0
	}
	return *l.cfg.TopKKNN
}

// decidirTopKKNN aplica la preselección si procede; ok=false si no procede (la pregunta sigue por el camino de siempre).
func (l *L3LLM) decidirTopKKNN(ctx context.Context, p Pregunta, previas Previas) (Respuesta, bool, error) {
	k := l.topKKNN()
	if k == 0 || p.Kind != KindChoice || len(p.Opciones) <= L3TopKKNNDesde || len(p.Opciones) <= k {
		return Respuesta{}, false, nil
	}
	prev, ok := previas["L2"]
	if !ok || prev.RankKNN == nil || len(prev.Probs) != len(p.Opciones) {
		return Respuesta{}, false, nil
	}
	probs := make([]float64, len(p.Opciones))
	cent := make([]float64, len(p.Opciones))
	for i, o := range p.Opciones {
		pv, ok1 := prev.Probs[o.ID]
		cv, ok2 := prev.RankKNN[o.ID]
		if !ok1 || !ok2 {
			return Respuesta{}, false, nil
		}
		probs[i], cent[i] = pv, cv
	}
	sel := RankTopK(probs, cent, k)
	sub := p
	sub.Opciones = make([]Opcion, len(sel))
	for i, j := range sel {
		sub.Opciones[i] = p.Opciones[j]
	}
	r, err := l.Decidir(ctx, sub)
	if err != nil || r.Probs == nil {
		return r, true, err
	}
	todas := make(map[string]float64, len(p.Opciones))
	for _, o := range p.Opciones {
		todas[o.ID] = r.Probs[o.ID] // fuera de las top_k: 0
	}
	r.Probs = todas
	return r, true, nil
}
