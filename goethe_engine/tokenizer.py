"""Text <-> token ids for the server. ByteTokenizer needs nothing (dev, tests, the CPU reference model); HFTokenizer
wraps a Hugging Face tokenizer directory through transformers (optional dependency: pip install goethe-engine[serve])."""


class ByteTokenizer:
    """Ids 0..255 are bytes; 256 is the end token."""

    vocab_size, eos_ids = 257, (256,)

    def encode(self, text):
        return list(text.encode())

    def chat(self, messages):
        return self.encode("".join(f"<|{m['role']}|>{m['content']}\n" for m in messages) + "<|assistant|>")

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")


class HFTokenizer:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(path)
        e = self.t.eos_token_id
        self.vocab_size, self.eos_ids = len(self.t), (() if e is None else (e,))

    def encode(self, text):
        return self.t.encode(text, add_special_tokens=False)

    def chat(self, messages):
        if not getattr(self.t, "chat_template", None):
            raise ValueError("the tokenizer has no chat template: use /v1/completions")
        r = self.t.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
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
