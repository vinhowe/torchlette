"""Minimal GPT-2/Qwen3 byte-level BPE detokenizer from tokenizer.json."""
import json, sys, re

TOK = "ckpts/qwen3-1.7b/tokenizer.json"

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def main():
    tk = json.load(open(TOK))
    vocab = tk["model"]["vocab"]  # token -> id
    id2tok = {v: k for k, v in vocab.items()}
    # added tokens (special)
    for at in tk.get("added_tokens", []):
        id2tok[at["id"]] = at["content"]
    dec = bytes_to_unicode()
    def detok(ids):
        s = "".join(id2tok.get(i, "") for i in ids)
        out = bytearray()
        for ch in s:
            if ch in dec:
                out.append(dec[ch])
            else:
                out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        ids = json.loads(line)
        print(repr(detok(ids)))

if __name__ == "__main__":
    main()
