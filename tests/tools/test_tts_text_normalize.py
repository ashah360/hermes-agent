from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from tools.tts_text_normalize import prepare_spoken_text


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        raise AssertionError("not used")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def test_prepare_spoken_text_expands_celsius_and_weather_units():
    raw = """## Christchurch today\n\n- **Now:** about **14°C**, feels like **14°C**\n- **Wind:** 9 km/h\n- **Rain:** 1.3 mm\n- **Range:** 11\u201317°C\n"""

    spoken = prepare_spoken_text(raw)

    assert "##" not in spoken
    assert "**" not in spoken
    assert "14 degrees Celsius" in spoken
    assert "11 to 17 degrees Celsius" in spoken
    assert "9 kilometres per hour" in spoken
    assert "1.3 millimetres" in spoken
    assert "°C" not in spoken
    assert "km/h" not in spoken


def test_prepare_spoken_text_polish_edge_cases():
    # Heading folds into the next sentence as a lead-in, not a bare label.
    assert prepare_spoken_text("## Weather\nIt will be sunny") == "Weather, It will be sunny."
    # Bare degree unit (no leading number) still expands.
    assert "degrees Celsius" in prepare_spoken_text("measured in °C")
    # Trailing comma is not swallowed into the amount.
    assert "300 US dollars" in prepare_spoken_text("US$300, next")
    # Real numeric rates expand, but and/or, N/A, IDs and dates are left intact.
    assert "5 dollars per month" in prepare_spoken_text("$5/month")
    assert "and/or" in prepare_spoken_text("choose and/or option")
    assert "N/A" in prepare_spoken_text("status N/A here")
    assert "2026/06/02" in prepare_spoken_text("due 2026/06/02 ok")


def test_prepare_spoken_text_expands_financial_magnitude_suffixes():
    # Regression suite from the live Jeeves voice patch: compact business
    # magnitudes must be spoken as words on every scripted speech path.
    assert prepare_spoken_text("$38M") == "38 million dollars"
    assert prepare_spoken_text("38M deposits") == "38 million deposits"
    assert prepare_spoken_text("$2.4B") == "2.4 billion dollars"
    assert prepare_spoken_text("750K") == "750 thousand"
    assert prepare_spoken_text("€38M") == "38 million euros"


def test_prepare_spoken_text_preserves_explicit_metric_metres():
    # Lowercase m followed by metric context stays metres (live patch guard).
    assert prepare_spoken_text("38m distance") == "38 metres distance"


def test_financial_magnitudes_do_not_break_existing_units():
    # The magnitude rule must not swallow genuine metric units.
    assert "120 millimetres" in prepare_spoken_text("rain was 120mm today")
    assert "9 kilometres per hour" in prepare_spoken_text("wind 9 km/h")
