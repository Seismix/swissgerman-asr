"""Model registry, audio preparation, and the two decoding backends."""
import hashlib
import pathlib
import subprocess

from transcript import Seg

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"

# One model, not a bench. Already in CTranslate2 form on the hub, so there is no
# build step: faster-whisper pulls 1.6 GB on first run and caches it.
#
# CC-BY-NC-4.0. Fine for coursework and personal research, NOT for commercial
# work - docs/licensing.md has the chain and the Apache-2.0 route if you need
# one. docs/findings.md has the measurements that chose it.
DEFAULT_MODEL = "OSTswiss/whisper-large-v3-turbo-swiss-german-ct2"


def parse_model_spec(spec=None):
    """-> (repo, key). A local directory, an HF repo id, or a hf.co URL.

    `key` is the last path/repo segment and names the output files, so the
    default writes out/whisper-large-v3-turbo-swiss-german-ct2.txt. Long, but
    derived by a rule rather than a lookup table, so it stays true for --model.
    """
    spec = str(spec or DEFAULT_MODEL).strip()
    if "://" in spec:
        host, _, path = spec.split("://", 1)[1].partition("/")
        if "huggingface.co" not in host:
            raise ValueError(f"only huggingface.co URLs are understood: {spec}")
        parts = [x for x in path.split("/") if x]
        for cut in ("tree", "blob", "resolve"):
            if cut in parts:
                parts = parts[:parts.index(cut)]
        if len(parts) != 2:
            raise ValueError(f"cannot read a repo id out of: {spec}")
        return "/".join(parts), parts[-1]
    path = pathlib.Path(spec).expanduser()
    if path.is_absolute() or spec.startswith((".", "/", "~")) or path.exists():
        return str(path.resolve()), path.name
    if spec.count("/") != 1:
        raise ValueError(f"expected 'Owner/Name', a URL, or a local path: {spec}")
    return spec, spec.split("/")[-1]


def detect_backend(repo):
    """CTranslate2 if it looks like a converted model, transformers otherwise.

    A CTranslate2 model always has model.bin at its root; a transformers
    checkpoint never does - it has *.safetensors. That test works on the hub as
    well as on disk, so a repo that is *already* converted can be used straight
    from HF with no local build step.

    The cache is consulted before the network so this still answers offline once
    the model has been pulled. If neither works we say "hf", which is the safe
    wrong answer: the transformers backend refuses a CTranslate2 repo loudly,
    where faster-whisper on a transformers repo fails deeper in. --backend
    overrides either way.
    """
    path = pathlib.Path(repo)
    if path.is_dir():
        return "fw" if (path / "model.bin").exists() else "hf"
    try:
        from huggingface_hub import list_repo_files, try_to_load_from_cache
        if try_to_load_from_cache(repo, "model.bin"):
            return "fw"
        return "fw" if "model.bin" in list_repo_files(repo) else "hf"
    except Exception:
        return "hf"


# Everything below assumed a CUDA box. It is resolved once at startup instead,
# so the pipeline runs on a machine without an NVIDIA GPU (slowly) rather than
# dying at model load.
COMPUTE_TYPES = {
    "cuda": "int8_float16",   # 8 GB cards: least VRAM at no measured quality cost
    "cpu": "int8",            # float16 on CPU is emulated and much slower
}


# What to say when a Radeon turns up and the CTranslate2 backend cannot use it.
ROCM_HELP = (
    "CTranslate2 has no AMD backend - it takes cuda and cpu, and rejects both "
    "rocm and hip - so no converted model can run on a Radeon.\n"
    "The transformers backend can, because it is only torch:\n"
    "  python run.py AUDIO --model Flix-AI/flix-swissgerman-full --longform\n"
    "That model is Apache-2.0, so the default's non-commercial restriction does "
    "not apply to it either. See docs/licensing.md.")


def gpu_kind():
    """'cuda', 'rocm' or None - what torch can see, not what CT2 can use.

    A ROCm build of torch reuses the whole torch.cuda namespace. On a Radeon,
    torch.cuda.is_available() is True and get_device_name(0) says "AMD Radeon",
    so believing it printed "device: cuda (AMD Radeon ...)" and then handed
    "cuda" to CTranslate2 - which fails at model load, one line after the tool
    announced a GPU. torch.version.hip is the only thing that separates the two
    builds, and it is unset on a real CUDA one.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def resolve_device(name="auto", backend="fw"):
    """-> (asr_device, torch_device). The same string except on AMD.

    ROCm runs everything torch-side, so a Radeon still does the ECAPA
    embeddings, and still does transcription on the transformers backend. Only
    CTranslate2 has to fall back. Returning one device for both is what would
    make that fallback cost diarization its GPU as well, for no reason.

    "rocm" is accepted as a --device spelling because it is the obvious thing
    to try on an AMD box; torch itself only answers to "cuda".
    """
    if name == "rocm":
        return "cuda", "cuda"
    if name != "auto":
        return name, name
    kind = gpu_kind()
    if kind is None:
        return "cpu", "cpu"
    if kind == "rocm" and backend == "fw":
        return "cpu", "cuda"
    return "cuda", "cuda"


def describe_device(device, torch_device=None):
    """One header line: what is about to run where, and why if it is not obvious."""
    kind = gpu_kind()
    if device == "cuda":
        import torch
        return f"{'rocm' if kind == 'rocm' else 'cuda'} ({torch.cuda.get_device_name(0)})"
    if kind == "rocm":
        import torch
        also = " Diarization still runs on the GPU." if torch_device == "cuda" else ""
        return (f"cpu - {torch.cuda.get_device_name(0)} is visible through ROCm, "
                f"but CTranslate2 has no AMD backend.{also}")
    return "cpu (no GPU found - this will be slow)"


def parse_time(s):
    """'93', '1:33' or '01:33.5' -> seconds."""
    parts = s.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {s}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def parse_clip(s):
    """'5:00-7:00' -> (300.0, 420.0). Either side may be empty."""
    if "-" not in s:
        raise ValueError(f"--clip wants START-END, got: {s}")
    a, b = s.split("-", 1)
    start = parse_time(a) if a.strip() else 0.0
    end = parse_time(b) if b.strip() else None
    if end is not None and end <= start:
        raise ValueError(f"--clip end must be after start: {s}")
    return start, end


def source_id(src: pathlib.Path, clip=None) -> dict:
    """What makes decoded audio reusable: which bytes, and which span of them.

    Keying a cache on the filename alone is not enough. `raw/talk.m4a` and
    `clean/talk.m4a` share a stem, as do `talk.m4a` and `talk.wav`, and a source
    re-exported in place keeps its name while its contents change. Any of those
    served another recording's audio from cache, silently.
    """
    st = src.stat()
    return {"path": str(src.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "clip": [clip[0], clip[1]] if clip else None}


def _digest(ident: dict) -> str:
    raw = f"{ident['path']}|{ident['size']}|{ident['mtime_ns']}|{ident['clip']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def to_wav16k(src: pathlib.Path, clip=None) -> pathlib.Path:
    """Decode to 16 kHz mono PCM once, into cache/, never beside the source.

    The source is often in a synced folder; a stray 48 MB WAV beside it syncs,
    and on a shared folder shows up for other people.
    """
    CACHE.mkdir(exist_ok=True)
    trim = []
    tag = ""
    if clip:
        start, end = clip
        tag = f".{start:g}-{end:g}" if end is not None else f".{start:g}-"
        trim = ["-ss", str(start)] + (["-to", str(end)] if end is not None else [])
    dst = CACHE / f"{src.stem}{tag}.{_digest(source_id(src, clip))}.16k.wav"
    if dst.exists():
        return dst

    # Decode to .part and rename on success. `ffmpeg -y` creates its output
    # before it can fail, so a decode killed partway (truncated source, full
    # disk, Ctrl-C) used to leave a short wav that `dst.exists()` accepted for
    # every later run - half a recording transcribed, timestamps looking normal,
    # nothing raised. os.replace is atomic within a filesystem, and cache/ is.
    print(f"[ffmpeg] {src.name}{tag or ''} -> {dst.name}")
    part = dst.with_name(dst.name + ".part")
    r = subprocess.run(["ffmpeg", "-y", *trim, "-i", str(src), "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav",
                        str(part)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        tail = "\n    ".join(r.stderr.strip().splitlines()[-4:])
        raise RuntimeError(f"ffmpeg could not decode {src}:\n    {tail}")
    part.replace(dst)
    return dst


def run_fw(repo, wav, offset=0.0, device="cuda", compute_type=None):
    """CTranslate2 backend. VAD-segmented, so segments follow speech pauses."""
    from faster_whisper import WhisperModel
    m = WhisperModel(repo, device=device,
                     compute_type=compute_type or COMPUTE_TYPES[device])
    segs, _ = m.transcribe(str(wav), language="de", task="transcribe",
                           beam_size=5, vad_filter=True,
                           condition_on_previous_text=False)
    return [Seg(s.start + offset, s.end + offset, s.text.strip()) for s in segs]


def run_hf(repo, wav, longform=False, offset=0.0, device="cuda"):
    """transformers backend. Slower, more VRAM, but no conversion step.

    Takes no compute_type: precision here follows the device, and accepting the
    argument only to drop it made `--compute-type int8` look like it had been
    applied while the run and its recorded timing were fp16. See
    `unsupported_compute_type`.
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    # fp16 matmuls are not implemented for most CPU kernels, so CPU gets fp32.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    proc = AutoProcessor.from_pretrained(repo)
    # Whisper's tokenizer is BPE. The WordPiece "cleanup" step deletes the space
    # *before* punctuation. Some fine-tunes ship it enabled; upstream does not.
    proc.tokenizer.clean_up_tokenization_spaces = False
    # Decoding params belong on the generation_config. Passing them as generate
    # kwargs as well is deprecated and warns on every call.
    gc = model.generation_config
    gc.language, gc.task, gc.num_beams = "de", "transcribe", 5
    # Manual 30 s chunking is what transformers warns about. --longform uses
    # Whisper's own sequential algorithm instead: faster, and it returns
    # sentence-level segments rather than one wall of text.
    chunking = {} if longform else dict(chunk_length_s=30, stride_length_s=5)
    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
                    dtype=dtype, device=device,
                    return_timestamps=True, **chunking)
    r = pipe(str(wav))
    if not r.get("chunks"):
        return [Seg(offset, None, r["text"].strip())]
    return [Seg(c["timestamp"][0] + offset,
                None if c["timestamp"][1] is None else c["timestamp"][1] + offset,
                c["text"].strip())
            for c in r["chunks"]]


def transcribe(repo, backend, wav, longform=False, offset=0.0, device="cuda",
               compute_type=None):
    if backend == "fw":
        return run_fw(repo, wav, offset, device, compute_type)
    assert not compute_type, "run.py should have rejected this up front"
    return run_hf(repo, wav, longform, offset, device)


def missing_local(repo):
    """A local-path model that has not been built yet.

    Tested by absolute-vs-relative, not by looking for a slash: this used to
    resolve a local default under HERE, so the repo always contained slashes and
    the old `"/" not in repo` check never fired - a missing directory fell
    through to the backend and surfaced as a decoder exception rather than a
    sentence about the path. The default is a remote repo now, so this only
    fires for a --model path the user passed.
    """
    path = pathlib.Path(repo)
    return path.is_absolute() and not path.exists()
