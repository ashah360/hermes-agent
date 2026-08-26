from hermes_state import SessionDB
from plugins.platforms.photon.conversation_actions import resolve_target_message_id


def _agent_with_db(db, session_id):
    """Minimal AIAgent stand-in wired to the REAL turn-boundary flush."""
    import run_agent as ra

    agent = ra.AIAgent.__new__(ra.AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._pending_cli_user_message = None
    agent._session_persist_lock = None
    return agent


def test_separate_turns_persist_their_own_inbound_ids_through_agent_flush(tmp_path):
    """Live regression: three successive completed turns all resolved
    ``target={trigger: true}`` to the FIRST turn's provider id, and every
    user row in state.db had ``platform_message_id`` NULL.

    The gateway skips its own DB write when the agent persists
    (``skip_db=agent_persisted``), so the ONLY path that can land the inbound
    provider id (``persist_user_message_id`` → ``user_msg["message_id"]``,
    see build_turn_context) in ``platform_message_id`` is the agent's
    ``_flush_messages_to_session_db``. This walks the real flush once per
    completed turn — no mid-turn steering — and asserts each turn's trigger
    resolves to its OWN exact inbound id against the persisted transcript.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("photon-session", source="photon")
    agent = _agent_with_db(db, "photon-session")

    inbound = [
        ("spc-msg-88ba9a91", "Ooooooooooooh"),
        ("spc-msg-2222", "WOW"),
        ("spc-msg-3333", "HYPE HYPE HYPE"),
    ]
    messages = []
    for provider_id, text in inbound:
        # Exactly what build_turn_context stages for a gateway turn that
        # carries persist_user_message_id (commit c96c480186).
        messages.append({"role": "user", "content": text, "message_id": provider_id})
        messages.append({"role": "assistant", "content": "!!"})
        assert agent._flush_messages_to_session_db(messages) is True

        target, error = resolve_target_message_id(
            session_id="photon-session",
            trigger_message_id=provider_id,
            messages_back=0,
            db=db,
        )
        assert (target, error) == (provider_id, None)

    # Every persisted user row carries its own exact inbound provider id —
    # the incident had all three NULL.
    rows = db.get_messages_as_conversation("photon-session")
    persisted = [m.get("message_id") for m in rows if m["role"] == "user"]
    assert persisted == [provider_id for provider_id, _ in inbound]

    # Relative targeting walks the REAL persisted ids, not the first turn's.
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="spc-msg-3333",
        messages_back=1,
        db=db,
    ) == ("spc-msg-2222", None)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="spc-msg-3333",
        messages_back=2,
        db=db,
    ) == ("spc-msg-88ba9a91", None)
    db.close()


def test_resolves_exact_trigger_and_earlier_persisted_user_bubbles(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("photon-session", source="photon")
    db.append_message(
        "photon-session", "user", "first", platform_message_id="imsg-first"
    )
    db.append_message("photon-session", "assistant", "answer")
    db.append_message(
        "photon-session", "user", "middle", platform_message_id="imsg-middle"
    )
    db.append_message(
        "photon-session", "user", "trigger", platform_message_id="imsg-trigger"
    )
    db.close()

    # Reopen to prove resolution uses persisted identity rather than adapter
    # memory from the process that received the messages.
    restarted = SessionDB(db_path=db_path, read_only=True)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="imsg-trigger",
        messages_back=0,
        db=restarted,
    ) == ("imsg-trigger", None)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="imsg-trigger",
        messages_back=1,
        db=restarted,
    ) == ("imsg-middle", None)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="imsg-trigger",
        messages_back=2,
        db=restarted,
    ) == ("imsg-first", None)
    restarted.close()


def test_debounced_turn_constituents_survive_agent_flush_for_messages_back(tmp_path):
    """A debounced multi-bubble turn persists ONE user row whose
    display_metadata carries the ordered constituent provider ids (what
    build_turn_context stamps from persist_user_display_metadata). After the
    real agent flush — i.e. across turn completion or a restart —
    messages_back must address the individual bubbles, then continue into
    earlier persisted turns, without exposing extra user rows.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("photon-session", source="photon")
    agent = _agent_with_db(db, "photon-session")

    messages = [
        {"role": "user", "content": "earlier", "message_id": "spc-earlier"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": "A\nB\nC",
            "message_id": "spc-C",
            "display_metadata": {
                "constituent_message_ids": ["spc-A", "spc-B", "spc-C"]
            },
        },
        {"role": "assistant", "content": "!!"},
    ]
    assert agent._flush_messages_to_session_db(messages) is True

    # Exactly two user rows persisted — the combined turn is ONE row.
    rows = db.get_messages_as_conversation("photon-session")
    assert [m["role"] for m in rows] == ["user", "assistant", "user", "assistant"]

    # Next turn D (live anchor [spc-D]) walks its own trigger, then the
    # persisted constituents newest-first, then the prior turn.
    expectations = ["spc-D", "spc-C", "spc-B", "spc-A", "spc-earlier"]
    for back, expected in enumerate(expectations):
        assert resolve_target_message_id(
            session_id="photon-session",
            trigger_message_id="spc-D",
            messages_back=back,
            db=db,
            turn_constituents=["spc-D"],
        ) == (expected, None)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="spc-D",
        messages_back=len(expectations),
        db=db,
        turn_constituents=["spc-D"],
    )[0] is None

    # Env-fallback path (no live anchor, e.g. post-restart tooling): the
    # persisted trigger row expands into its constituents too.
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="spc-C",
        messages_back=1,
        db=db,
    ) == ("spc-B", None)
    assert resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="spc-C",
        messages_back=3,
        db=db,
    ) == ("spc-earlier", None)
    db.close()


def test_historical_bubble_without_platform_id_fails_closed(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("photon-session", source="photon")
    db.append_message("photon-session", "user", "legacy")
    db.append_message(
        "photon-session", "user", "trigger", platform_message_id="imsg-trigger"
    )

    target, error = resolve_target_message_id(
        session_id="photon-session",
        trigger_message_id="imsg-trigger",
        messages_back=1,
        db=db,
    )

    assert target is None
    assert "predates platform-id persistence" in error
    db.close()


def test_trigger_must_belong_to_active_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("current", source="photon")
    db.create_session("other", source="photon")
    db.append_message("current", "user", "here", platform_message_id="current-id")
    db.append_message("other", "user", "there", platform_message_id="other-id")

    target, error = resolve_target_message_id(
        session_id="current",
        trigger_message_id="other-id",
        messages_back=1,
        db=db,
    )

    assert target is None
    assert "not present in this conversation" in error
    db.close()
