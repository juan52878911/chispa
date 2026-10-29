//go:build ort

package main

import "github.com/juan52878911/chispa/server/decide"

func fijarRutaORT(lib string) { decide.RutaORT = lib }
