package tok

import (
	"sort"
	"strings"
)

// nfc normaliza a NFC (forma canónica compuesta) sin dependencias: el módulo no puede traer golang.org/x/text. Las tablas salen de
// gen_nfc.py (nfc_tablas.go); el Hangul es algorítmico. nfc_test.go lo compara con unicodedata.normalize('NFC') de Python.

const (
	hangulS0, hangulL0, hangulV0, hangulT0 = 0xAC00, 0x1100, 0x1161, 0x11A7
	hangulLN, hangulVN, hangulTN           = 19, 21, 28
	hangulNN                               = hangulVN * hangulTN
	hangulSN                               = hangulLN * hangulNN
)

func claseCombinacion(r rune) uint8 {
	if r < 0x300 {
		return 0
	}
	i := sort.Search(len(nfcCCC), func(i int) bool { return nfcCCC[i][1] >= int32(r) })
	if i < len(nfcCCC) && nfcCCC[i][0] <= int32(r) {
		return uint8(nfcCCC[i][2])
	}
	return 0
}

func descomponer(dst []rune, r rune) []rune {
	if r >= hangulS0 && r < hangulS0+hangulSN {
		s := r - hangulS0
		dst = append(dst, hangulL0+s/hangulNN, hangulV0+(s%hangulNN)/hangulTN)
		if t := s % hangulTN; t != 0 {
			dst = append(dst, hangulT0+t)
		}
		return dst
	}
	if d, ok := nfcDescomp[r]; ok {
		return append(dst, d...)
	}
	return append(dst, r)
}

func componerPar(a, b rune) (rune, bool) {
	if a >= hangulL0 && a < hangulL0+hangulLN && b >= hangulV0 && b < hangulV0+hangulVN {
		return hangulS0 + ((a-hangulL0)*hangulVN+(b-hangulV0))*hangulTN, true
	}
	if a >= hangulS0 && a < hangulS0+hangulSN && (a-hangulS0)%hangulTN == 0 && b > hangulT0 && b < hangulT0+hangulTN {
		return a + (b - hangulT0), true
	}
	c, ok := nfcComp[[2]rune{a, b}]
	return c, ok
}

// NFC devuelve s en forma NFC. s debe ser UTF-8 válido (lo que no lo es se deja pasar tal cual, como RuneError).
func NFC(s string) string {
	rapido := true
	for i := 0; i < len(s); i++ {
		if s[i] >= 0xCC { // primer byte de U+0300 en adelante (los de U+0080..U+02FF son 0xC2..0xCB)
			rapido = false
			break
		}
	}
	if rapido {
		return s
	}
	var rs []rune
	for _, r := range s {
		rs = descomponer(rs, r)
	}
	// orden canónico: dentro de cada racha de no iniciales, por clase (estable)
	for i := 0; i < len(rs); {
		if claseCombinacion(rs[i]) == 0 {
			i++
			continue
		}
		j := i
		for j < len(rs) && claseCombinacion(rs[j]) != 0 {
			j++
		}
		seg := rs[i:j]
		sort.SliceStable(seg, func(a, b int) bool { return claseCombinacion(seg[a]) < claseCombinacion(seg[b]) })
		i = j
	}
	// composición
	var out strings.Builder
	out.Grow(len(s))
	buf := make([]rune, 0, len(rs))
	ini := -1 // índice del último iniciador en buf
	var ultimaClase int
	for _, r := range rs {
		cc := int(claseCombinacion(r))
		if ini >= 0 {
			if c, ok := componerPar(buf[ini], r); ok && (ultimaClase < cc || ultimaClase == 0 && len(buf)-1 == ini) {
				// no bloqueado: nada entre el iniciador y r con clase ≥ cc (ultimaClase es la de lo último que quedó tras el iniciador)
				buf[ini] = c
				continue
			}
		}
		if cc == 0 {
			ini = len(buf)
			ultimaClase = 0
		} else {
			ultimaClase = cc
		}
		buf = append(buf, r)
	}
	for _, r := range buf {
		out.WriteRune(r)
	}
	return out.String()
}
