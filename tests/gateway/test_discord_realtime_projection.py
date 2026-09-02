"""Projection builder for the realtime lane (ADR D5).

The projection must be assembled from the SAME identity sources the full
agent uses — ``agent/prompt_builder.load_soul_md`` (profile SOUL.md) and the
USER.md profile block via ``MemoryStore.format_for_system_prompt("user")`` —
be byte-stable, budget-capped with deterministic section-boundary truncation,
and never include tool catalogs or the full prompt scaffold.

E2E against a temp HERMES_HOME (autouse fixture redirects it); the real
loaders run — no mocks.
"""

import os
from pathlib import Path

import pytest


def _hermes_home() -> Path:
    return Path(os.environ["HERMES_HOME"])


@pytest.fixture
def identity_fixture():
    home = _hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "SOUL.md").write_text(
        "# Jeeves\n\nYou are Jeeves, Shaan's executive agent. Dry wit, precise.\n"
        "Arman is the operator; Ghost handles deployments.\n"
        "\n## Style\n\nSpeak plainly. Numbers matter.\n",
        encoding="utf-8",
    )
    mem = home / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "USER.md").write_text(
        "- Shaan prefers totals in millions, not raw digits\n"
        "- Weekly review is Friday 9am Pacific\n",
        encoding="utf-8",
    )
    return home


def _build(**overrides):
    from plugins.platforms.discord.realtime.config import load_realtime_voice_config
    from plugins.platforms.discord.realtime.projection import build_projection

    kwargs = dict(
        config=load_realtime_voice_config({"enabled": True}),
        guild_name="Nous HQ",
        voice_channel_name="war-room",
        text_channel_name="jeeves-log",
        user_display_name="Shaan",
    )
    kwargs.update(overrides)
    return build_projection(**kwargs)


class TestProjectionSources:
    def test_identity_and_user_profile_from_real_helpers(self, identity_fixture):
        proj = _build()
        # SOUL.md identity present, loaded through load_soul_md.
        assert "You are Jeeves, Shaan's executive agent" in proj
        # USER.md stable preferences present, via the MemoryStore renderer.
        assert "totals in millions" in proj
        assert "Friday 9am Pacific" in proj

    def test_static_session_context_present(self, identity_fixture):
        proj = _build()
        assert "war-room" in proj
        assert "jeeves-log" in proj
        assert "Shaan" in proj

    def test_agency_tone_and_grounding_contracts_present(self, identity_fixture):
        proj = _build()
        low = proj.lower()
        # Agency: dispatch IS Jeeves's own hands — immediate, same turn.
        assert "you are jeeves" in low
        assert "your own hands" in low
        assert "immediately" in low
        assert "hermes_dispatch" in proj
        # Numeric grounding stays (correctness, not restriction).
        assert "never state business or data figures" in low
        # Tone contract: rudeness never changes whether the work happens.
        assert "profanity" in low
        assert "never lecture" in low
        # Style contract: no canned acks, no tool-name narration.
        assert "never use canned" in low
        assert "tool names" in low
        # SOUL identity passes through untouched (Arman/Ghost present).
        assert "Arman is the operator" in proj
        assert "Ghost handles deployments" in proj

    def test_contracts_contain_no_restriction_or_separate_entity_language(self):
        """Principal amendment: nothing is ever restricted, and the worker
        machinery is never framed as a separate entity. Asserted on the
        module contract constants (SOUL.md content is the user's own)."""
        from plugins.platforms.discord.realtime import projection as proj_mod

        contracts = " ".join([
            proj_mod._AGENCY_CONTRACT,
            proj_mod._AUTHORITY_CONTRACT,
            proj_mod._TONE_CONTRACT,
            proj_mod._STYLE_CONTRACT,
        ]).lower()
        for forbidden in ("approv", "permission", "cannot", "not allowed",
                          "blocked on", "background worker", "backend"):
            assert forbidden not in contracts, f"restriction language: {forbidden!r}"
        # And no Approvals section exists at all any more.
        assert not hasattr(proj_mod, "_APPROVAL_CONTRACT_AUTO_DENY")
        assert not hasattr(proj_mod, "_APPROVAL_CONTRACT_AUTO_APPROVE")

    def test_fallback_identity_without_soul_md(self):
        # No SOUL.md in the temp home → DEFAULT_AGENT_IDENTITY fallback.
        proj = _build()
        assert len(proj) > 200  # still a full projection
        assert "hermes_dispatch" in proj


class TestProjectionStability:
    def test_byte_stable_across_builds(self, identity_fixture):
        assert _build() == _build()

    def test_excludes_prompt_scaffold_and_tool_catalog(self, identity_fixture):
        proj = _build()
        for marker in ("skill_view", "delegate_task", "MEMORY.md", "terminal("):
            assert marker not in proj

    def test_budget_ceiling_with_section_boundary_truncation(self, identity_fixture):
        home = identity_fixture
        big_section = "## Bravo\n\n" + ("lorem ipsum " * 2000)
        (home / "SOUL.md").write_text(
            "# Jeeves\n\nYou are Jeeves, Shaan's executive agent.\n\n"
            "## Alpha\n\nShort and load-bearing.\n\n" + big_section,
            encoding="utf-8",
        )
        from plugins.platforms.discord.realtime.projection import PROJECTION_MAX_CHARS

        proj = _build()
        assert len(proj) <= PROJECTION_MAX_CHARS
        # Whole-section truncation: Alpha survives, Bravo is dropped entirely —
        # never a mid-section fragment.
        assert "Short and load-bearing." in proj
        assert "lorem ipsum" not in proj
