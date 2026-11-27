"""goethe-engine: serve | worker | turn | index verify|build | pack match|verify|show|need|runs|plan|build|sign|check."""
import sys

USAGE = """usage: goethe-engine serve [--host H --port P --context-length N --max-running-requests N --served-model-name S
                            --api-key K --pack FILE.gcpack [--overlay O]... --weights PATH... [--tokenizer DIR] [--device N]]
       goethe-engine worker --socket PATH [--device N]          (worker protocol v1, spec/proto/worker.proto)
       goethe-engine turn DIR                                   (what a model's turn holds: {"reasoning", "tools"})
       goethe-engine index verify --index F --sig F [--last-serial N]
       goethe-engine index build --pack F [--pack F]... --serial N --expires T --key PEM --out DIR [--index F --sig F] [--tier T] [--min-engine V]
       goethe-engine pack match --index F --sig F --model-dir DIR --arch sm_XX [--last-serial N] [--tier T]
       goethe-engine pack need PREFIX --model-dir DIR [--arch sm_XX --sku S --numerics ..]
       goethe-engine pack verify FILE [--sig F] [--overlay O]... [--arch sm_XX] [--partial --model-dir DIR] | pack show FILE [--sig F]
       goethe-engine pack runs FILE [--sig F]     (does the visible device run this pack: one JSON line; exit 0 yes, 3 no, 1 error)
       goethe-engine --version | --help"""


def worker_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="goethe-engine worker")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--device", type=int, default=0)
    return ap


def turn_parser():
    import argparse
    ap = argparse.ArgumentParser(prog="goethe-engine turn")
    ap.add_argument("dir", help="a Hugging Face tokenizer directory")
    return ap


def turn(d):
    """{"reasoning", "tools"}: the parts of the turn format the engine reads this tokenizer's output by (tokenizer.TURNS); false without one."""
    from .tokenizer import HFTokenizer
    t = HFTokenizer(d).turn
    kinds = {k for k, _ in (t or {}).get("parts", {}).values()}
    return dict(reasoning="reasoning" in kinds, tools="tool" in kinds)


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if a[:1] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if a[:1] == ["--version"]:
        from . import __version__
        print(__version__)
        return 0
    if a[:1] == ["serve"]:
        from . import serve
        return serve.main(a[1:]) or 0
    if a[:1] == ["turn"]:
        import json
        try:
            print(json.dumps(turn(turn_parser().parse_args(a[1:]).dir)))
        except (OSError, ValueError) as e:
            print(f"goethe-engine: turn: {e}", file=sys.stderr)
            return 1
        return 0
    if a[:1] == ["worker"]:
        from . import worker
        w = worker_parser().parse_args(a[1:])
        worker.serve(w.socket, w.device)
        return 0
    if a[:2] in (["index", "verify"], ["index", "build"]) or a[:2] == ["pack", "match"]:
        from . import index
        return index.main(a[1:])
    if a[:1] == ["pack"] and a[1:2] in (["verify"], ["show"], ["need"], ["runs"], ["plan"], ["build"], ["sign"], ["check"], ["-h"], ["--help"]):
        from . import pack
        pack.main(a[1:])
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
