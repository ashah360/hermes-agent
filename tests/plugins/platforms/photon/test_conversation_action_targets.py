from hermes_state import SessionDB
from plugins.platforms.photon.conversation_actions import resolve_target_message_id


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
