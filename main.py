# main.py

import os
import threading
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import asyncio

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables from .env
load_dotenv()

# Health-check HTTP server
health_app = FastAPI()

@health_app.get("/healthz")
async def healthz():
    return {"status": "ok"}

def run_health_server():
    port = int(os.environ.get("PORT", "8081"))
    uvicorn.run(health_app, host="0.0.0.0", port=port, log_level="warning")

class FitnessAssistant(Agent):
    """
    AndrofitAI: An energetic, voice-interactive, supportive AI personal gym coach.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach. "
                "Start every workout with 'How's your vibe today? Ready to crush it?' "
                "Prompt users for goals and equipment, then generate personalized workouts. "
                "Guide each rep and rest, support commands like 'Pause' or 'Skip,' "
                "and deliver motivational feedback throughout."
            )
        )

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="tts-1", voice="alloy", instructions="Speak friendly."),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(
        room=ctx.room,
        agent=FitnessAssistant(),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await session.generate_reply(instructions="Greet the user and offer assistance.")

if __name__ == "__main__":
    # Start health endpoint in background thread
    threading.Thread(target=run_health_server, daemon=True).start()

    # Start LiveKit agent worker (blocking)
    try:
        agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        print(f"Error starting agent: {e}")
        import sys
        sys.exit(1)
