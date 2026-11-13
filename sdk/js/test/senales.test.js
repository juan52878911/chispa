import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("index.d.ts declara los campos opcionales calibrada, truncado, tokens_originales y unk_pct", () => {
  const d = readFileSync(new URL("../src/index.d.ts", import.meta.url), "utf8");
  for (const c of ["calibrada?: boolean", "truncado?: boolean", "tokens_originales?: number", "unk_pct?: number"]) {
    assert.ok(d.includes(c), c);
  }
});
