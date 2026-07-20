from PIL import Image

from inspection.pipeline import InspectionPipeline


class FakeRunner:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, image, prompt, max_new_tokens=None):
        self.calls += 1
        return next(self.responses), 0.01


def test_verifier_rejects_unsupported_issue():
    first = (
        '{"result":"attention","issue_type":"floor_obstruction",'
        '"evidence":"A small object is on the floor.","suggestion":"Move it."}'
    )
    runner = FakeRunner([first, "no"])
    pipeline = InspectionPipeline(runner=runner)
    trace = pipeline.inspect(Image.new("RGB", (8, 8)), "verified")
    assert trace.parsed.label == "normal"
    assert trace.confirmation_decision == "no"
    assert runner.calls == 2


def test_normal_result_skips_verifier():
    first = '{"result":"normal","issue_type":"","evidence":"No issue.","suggestion":""}'
    runner = FakeRunner([first])
    pipeline = InspectionPipeline(runner=runner)
    trace = pipeline.inspect(Image.new("RGB", (8, 8)), "verified")
    assert trace.parsed.label == "normal"
    assert trace.confirmation_output == ""
    assert runner.calls == 1


def test_decomposed_checks_avoid_first_label_bias():
    runner = FakeRunner(
        [
            "no | the walking path is clear",
            "yes | many dirty bowls cover the counter",
            "no | none",
            "yes",
        ]
    )
    pipeline = InspectionPipeline(runner=runner)
    trace = pipeline.inspect(Image.new("RGB", (8, 8)), "decomposed")
    assert trace.parsed.label == "countertop_clutter"
    assert trace.confirmation_decision == "yes"
    assert runner.calls == 4
