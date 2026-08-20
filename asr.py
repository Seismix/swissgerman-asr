"""Model registry, audio preparation, and the two decoding backends."""
import pathlib
import subprocess

from transcript import Seg

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"

DEFAULT = "flix-ct2"

MODELS = {
    # key: (repo, backend, license, note)
    "flix-ct2": (str(HERE / "flix-ct2"), "fw", "Apache-2.0",
                 "RECOMMENDED. Flix-AI/flix-swissgerman-full converted to CTranslate2 fp16. "
                 "Run ./build_flix_ct2.sh once to create it. Best speed/quality/licence mix."),
    "flix": ("Flix-AI/flix-swissgerman-full", "hf", "Apache-2.0",
             "same weights via HuggingFace transformers. 25.6% WER / 13.8% cWER on "
             "ASGDTS (arXiv 2606.07608). Use --longform."),
    "ct2": ("OSTswiss/whisper-large-v3-turbo-swiss-german-ct2", "fw", "CC-BY-NC-4.0",
            "turbo fine-tune, 1.6 GB, fastest and lowest VRAM. NON-COMMERCIAL - the "
            "upstream weights are CC-BY-NC, so any repo labelling this Apache-2.0 is wrong."),
    "base-v3": ("openai/whisper-large-v3", "hf", "Apache-2.0",
                "control: no Swiss German fine-tune. 28.56% WER on ASGDTS, i.e. only "
                "~3 points behind the fine-tunes. Run it before assuming a FT is needed."),
}


# Everything below assumed a CUDA box. It is resolved once at startup instead,
# so the pipeline runs on a machine without an NVIDIA GPU (slowly) rather than
# dying at model load.
COMPUTE_TYPES = {
    "cuda": "int8_float16",   # 8 GB cards: least VRAM at no measured quality cost
    "cpu": "int8",            # float16 on CPU is emulated and much slower
}


def resolve_device(name="auto"):
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def describe_device(device):
    if device != "cuda":
        return "cpu (no CUDA GPU found - this will be slow)"
    import torch
    return f"cuda ({torch.cuda.get_device_name(0)})"


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


def to_wav16k(src: pathlib.Path, clip=None) -> pathlib.Path:
    """Decode to 16 kHz mono PCM once, into cache/, never beside the source.

    The clip range is part of the cache name, so two spans of one recording do
    not overwrite each other.
    """
    CACHE.mkdir(exist_ok=True)
    tag = ""
    trim = []
    if clip:
        start, end = clip
        tag = f".{start:g}-{end:g}" if end is not None else f".{start:g}-"
        trim = ["-ss", str(start)] + (["-to", str(end)] if end is not None else [])
    dst = CACHE / f"{src.stem}{tag}.16k.wav"
    if not dst.exists():
        print(f"[ffmpeg] {src.name}{tag or ''} -> {dst.name}")
        subprocess.run(["ffmpeg", "-y", *trim, "-i", str(src), "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
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


def run_hf(repo, wav, longform=False, offset=0.0, device="cuda",
           compute_type=None):
    """transformers backend. Slower, more VRAM, but no conversion step."""
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


def transcribe(key, wav, longform=False, offset=0.0, device="cuda",
               compute_type=None):
    repo, backend, _, _ = MODELS[key]
    if backend == "fw":
        return run_fw(repo, wav, offset, device, compute_type)
    return run_hf(repo, wav, longform, offset, device, compute_type)


def missing_local(key):
    """A local-path model that has not been built yet.

    Tested by absolute-vs-relative, not by looking for a slash: flix-ct2's repo
    is built from HERE, so it always contains slashes and the old
    `"/" not in repo` check never fired. An unbuilt flix-ct2 fell through to
    the backend and surfaced as a decoder exception instead of "run
    ./build_flix_ct2.sh".
    """
    repo, _, _, _ = MODELS[key]
    path = pathlib.Path(repo)
    return path.is_absolute() and not path.exists()
