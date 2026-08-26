"""Async registered tools must observe the CURRENT turn's session context.

Live Photon regression: three successive completed turns each called the
async ``photon_conversation_action`` tool with ``target={trigger: true}``,
and every call resolved ``HERMES_SESSION_MESSAGE_ID`` to the FIRST turn's
provider message id — three reactions landed on the first bubble.

The dispatch topology under test is the real gateway one:

    per-turn context (set_session_vars)             [event-loop task]
      -> ctx.run(...) on a REUSED executor thread   [agent loop]
        -> registry.execute(async tool)             [model_tools._run_async]
          -> per-thread persistent worker loop      [_get_worker_loop]

The worker loop and the executor thread are both created on the FIRST
invocation and reused for every later turn, so any context captured at
creation time instead of call time replays turn 1's identity forever.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from tools.registry import registry

TOOL_NAME = "_test_session_ctx_probe"


def _register_probe_tool():
    async def probe(_args, **_kwargs):
        return get_session_env("HERMES_SESSION_MESSAGE_ID", "")

    registry.register(
        name=TOOL_NAME,
        toolset="_test_session_ctx",
        schema={"name": TOOL_NAME, "description": "probe", "parameters": {}},
        handler=probe,
        is_async=True,
        override=True,
    )


def test_async_tool_observes_each_turns_message_id_on_reused_worker_thread():
    _register_probe_tool()

    # One worker thread reused across every turn — the gateway executor's
    # steady state (thread_name_prefix="hermes-gateway", threads recycled).
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-gateway-test")
    worker_ids = []
    observed = []

    def _turn(message_id: str) -> None:
        """One completed turn: bind session vars, dispatch the async tool."""
        tokens = set_session_vars(
            platform="photon",
            chat_id="space-1",
            session_key="agent:main:photon:dm:space-1:user-1",
            message_id=message_id,
        )
        try:

            def _agent_loop():
                worker_ids.append(threading.get_ident())
                return registry.dispatch(TOOL_NAME, {})

            # _run_in_executor_with_context: context copy handed to the
            # (reused) executor thread via ctx.run.
            ctx = copy_context()
            observed.append(executor.submit(ctx.run, _agent_loop).result(timeout=30))
        finally:
            clear_session_vars(tokens)

    try:
        for message_id in ("spc-msg-1", "spc-msg-2", "spc-msg-3"):
            _turn(message_id)
    finally:
        executor.shutdown(wait=False)
        registry.deregister(TOOL_NAME)

    # Same OS thread (and therefore the same persistent worker loop) served
    # every call — the reuse scenario the regression depends on.
    assert len(set(worker_ids)) == 1
    assert observed == ["spc-msg-1", "spc-msg-2", "spc-msg-3"]
