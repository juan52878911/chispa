package tok

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"
)

func TestNFCContraUnicodedata(t *testing.T) {
	f, err := os.Open("testdata/nfc_casos.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	n, cambian := 0, 0
	for sc.Scan() {
		var par [2]string
		if err := json.Unmarshal(sc.Bytes(), &par); err != nil {
			t.Fatal(err)
		}
		n++
		if par[0] != par[1] {
			cambian++
		}
		if got := NFC(par[0]); got != par[1] {
			t.Fatalf("caso %d: NFC(%+q) = %+q, quiero %+q", n, par[0], got, par[1])
		}
	}
	if n < 4000 || cambian < 1500 {
		t.Fatalf("pocos casos: %d (%d cambian)", n, cambian)
	}
	t.Logf("%d casos, %d cambian, Unicode %s", n, cambian, versionUnicodeNFC)
}
