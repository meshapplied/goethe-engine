"""goethe-engine: serve | worker | index verify|build | pack match|verify|show|need|runs|plan|build."""
import sys

USAGE = """usage: goethe-engine serve [--host H --port P --context-length N --max-running-requests N --served-model-name S
                            --api-key K --pack FILE.gcpack [--overlay O]... --weights PATH... [--tokenizer DIR] [--device N]]
       goethe-engine worker --socket PATH [--device N]          (worker protocol v1, spec/proto/worker.proto)
       goethe-engine index verify --index F --sig F [--last-serial N]
       goethe-engine index build --pack F [--pack F]... --serial N --expires T --key PEM --out DIR [--tier T] [--min-engine V]
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
    if a[:1] == ["worker"]:
        from . import worker
        w = worker_parser().parse_args(a[1:])
        worker.serve(w.socket, w.device)
        return 0
    if a[:2] in (["index", "verify"], ["index", "build"]) or a[:2] == ["pack", "match"]:
        from . import index
        return index.main(a[1:])
    if a[:1] == ["pack"] and a[1:2] in (["verify"], ["show"], ["need"], ["runs"], ["plan"], ["build"], ["-h"], ["--help"]):
        from . import pack
        pack.main(a[1:])
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
