"""Text <-> token ids for the server. ByteTokenizer needs nothing (dev, tests, the CPU reference model); HFTokenizer
wraps a Hugging Face tokenizer directory through transformers (optional dependency: pip install goethe-engine[serve])."""
import json
import os

# Turn formats, one row per model family: a vocabulary holding a row's markers as single tokens reads its output by that row.
TURNS = (dict(head="<|message_model|>", end="<|end_message|>",
              parts={"<|content_thinking|>": "reasoning", "<|content_text|>": "content", "<|content_invoke_tool_json|>": "tool"}),)


class ByteTokenizer:
    """Ids 0..255 are bytes; 256 is the end token."""

    vocab_size, eos_ids, turn = 257, (256,), None

    def encode(self, text):
        return list(text.encode())

    def embed_input(self, text):
        return self.encode(text)

    def chat(self, messages, tools=None, **kw):
        if tools or kw:
            raise ValueError("the byte tokenizer has no chat template for tools or template options")
        return self.encode("".join(f"<|{m['role']}|>{m['content']}\n" for m in messages) + "<|assistant|>")

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


def model_stop_ids(path):
    """The stop ids the model directory declares: generation_config.json's eos_token_id, else config.json's (an int or a
    list, e.g. a chat model's end of turn next to its end of text). Empty when neither names any."""
    for name in ("generation_config.json", "config.json"):
        try:
            with open(os.path.join(path, name)) as f:
                v = json.load(f).get("eos_token_id")
        except (OSError, ValueError, AttributeError):
            continue
        ids = [v] if isinstance(v, int) else [x for x in v if isinstance(x, int)] if isinstance(v, list) else []
        if ids:
            return ids
    return []


class HFTokenizer:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)
        e = self.t.eos_token_id
        # A generation ends on the tokenizer's eos and on every stop id the model declares.
        ids = ([] if e is None else [e]) + model_stop_ids(path)
        self.vocab_size, self.eos_ids = len(self.t), tuple(dict.fromkeys(ids))
        tid = {s: self.t.convert_tokens_to_ids(s) for r in TURNS for s in (r["head"], r["end"], *r["parts"])}
        ok = lambda s: isinstance(tid[s], int) and tid[s] != self.t.unk_token_id
        self.turn = next((dict(head=tid[r["head"]], end=tid[r["end"]], parts={tid[k]: v for k, v in r["parts"].items()})
                          for r in TURNS if all(map(ok, (r["head"], r["end"], *r["parts"])))), None)

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def embed_input(self, text):
        """An embedding model's input: the text with the tokenizer's special tokens (cls / sep, bos / eos) as the model was trained."""
        return self.t.encode(text, add_special_tokens=True)

    def chat(self, messages, tools=None, **kw):
        if not getattr(self.t, "chat_template", None):
            raise ValueError("the tokenizer has no chat template: use /v1/completions")
        from jinja2 import TemplateError
        try:
            r = self.t.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True, **kw)
        except TemplateError as e:
            raise ValueError(f"the chat template refused the request: {e}")
        return list(r["input_ids"] if hasattr(r, "keys") else r)   # transformers 5 returns a mapping

    def decode(self, ids):
        return self.t.decode(ids, skip_special_tokens=True)


class TextStream:
    """Incremental detokenizer: feed(ids) -> the new text; a trailing partial character is held back."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def feed(self, new):
        self.ids += new
        old = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        if len(cur) > len(old) and not cur.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return cur[len(old):]
        return ""

    def flush(self):
        old = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return cur[len(old):]


class Turn:
    """Turn ids as (reasoning | content, text) pieces and whole (tool, call) parts; a call that does not parse stays content. O(1) per id."""

    def __init__(self, tok, fmt=None):
        self.tok, self.fmt = tok, fmt
        self._reset()

    def _reset(self):
        self.kind, self.head, self.ts, self.call = None if self.fmt else "content", [], TextStream(self.tok), ""

    def feed(self, ids, final=False):
        out, f = [], self.fmt
        for t in ids:
            if f and (t == f["head"] or t == f["end"] or t in f["parts"]):
                if t in f["parts"] and self.kind is None:
                    self.head = []
                self._close(out)
                self.kind = f["parts"].get(t)
            elif self.kind is None:
                self.head.append(t)
            else:
                self._put(out, self.ts.feed([t]))
        if final:
            self._close(out)
        return [p for p in out if p[1]]

    def _put(self, out, s):
        if self.kind == "tool":
            self.call += s
        else:
            out.append((self.kind, s))

    def _close(self, out):
        if self.kind is None:
            out.append(("content", self.tok.decode(self.head)))
        else:
            self._put(out, self.ts.flush())
        if self.kind == "tool":
            try:
                j = json.loads(self.call)
                ok = isinstance(j, dict) and isinstance(j.get("name"), str) and isinstance(j.get("args", {}), dict)
            except ValueError:
                ok = False
            out.append(("tool", dict(name=j["name"], arguments=json.dumps(j.get("args", {}), ensure_ascii=False))) if ok else ("content", self.call))
        self._reset()
