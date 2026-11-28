"""Read the existing native render's meter values without another measurement."""

from contextlib import contextmanager


def native_print_values(settings, metrics):
    from negpy.features.exposure.auto_sliders import print_shown_values
    from negpy.features.exposure.papers import effective_paper_profile

    shown = print_shown_values(settings.exposure, settings.process.process_mode, metrics)
    return dict(
        paper=effective_paper_profile(settings.exposure.paper_profile,
                                      settings.process.process_mode).label,
        density=float(shown.get("density", settings.exposure.density)),
        grade=float(shown.get("grade", settings.exposure.grade)),
        auto_density="density" in shown, auto_grade="grade" in shown,
    )


@contextmanager
def collect_print_values(processor):
    processor.pure_native_print = None
    if not hasattr(processor, "run_pipeline"):
        yield
        return
    original = processor.run_pipeline
    previous = vars(processor).get("run_pipeline")

    def run(*args, **kwargs):
        pixels, metrics = original(*args, **kwargs)
        settings = args[1] if len(args) > 1 else kwargs.get("settings")
        if settings is not None and metrics:
            processor.pure_native_print = native_print_values(settings, metrics)
        return pixels, metrics

    processor.run_pipeline = run
    try:
        yield
    finally:
        if previous is None:
            del processor.run_pipeline
        else:
            processor.run_pipeline = previous


def print_caption(summary):
    if not summary:
        return ""
    return (f" · {summary['paper']} · 印相密度 {summary['density']:.2f}"
            + ("（自动）" if summary['auto_density'] else "")
            + f" · ISO-R {summary['grade']:.0f}"
            + ("（自动）" if summary['auto_grade'] else ""))
