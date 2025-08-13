import nest_asyncio
nest_asyncio.apply()  # Allow nested event loops

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
import os
import asyncio

# Load environment variables
load_dotenv()

# Try to import turn detector with fallback
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    print("Turn detector not available - using VAD only")
    TURN_DETECTOR_AVAILABLE = False

class FitnessAssistant(Agent):
    """AndrofitAI: An energetic, voice-interactive AI personal gym coach."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach. "
                "Start every workout session with a warm, personal greeting like 'How's your vibe today? Ready to crush it?' "
                "Prompt users to share their fitness goals, experience level, available equipment, and time, then dynamically generate customized workout plans — "
                "Guide workouts in real time with step-by-step verbal instructions, providing clear cues for each exercise, set, rep, and rest interval — "
                "Support voice commands like 'Pause,' 'Skip,' or 'Make it easier' to ensure users feel in control. "
                "Consistently deliver motivational, context-aware feedback—if a user expresses fatigue, reassure them with, 'You're tough, just two more!' "
                "Share essential form and technique tips by describing correct posture and alignment, and confidently answer questions like 'How's a deadlift done?' "
                "Adopt an authentic personal trainer style: build rapport with empathetic, conversational exchanges and respond to user mood or progress. "
                "During rest intervals, initiate brief, engaging fitness discussions—for example, 'Protein aids recovery; try eggs post-workout.' "
                "Accurately count reps using user grunts, or offer a motivating cadence to keep users on pace, cheering them through every set. "
                "Always focus on making each session positive, safe, goal-oriented, and truly personalized."
            )
        )

async def entrypoint(ctx: agents.JobContext):
    try:
        # Configure turn detection
        if TURN_DETECTOR_AVAILABLE:
            turn_detection = MultilingualModel()
            print("Using multilingual turn detection model")
        else:
            turn_detection = "vad"
            print("Using VAD-only turn detection")

        session = AgentSession(
            stt=openai.STT(model="whisper-1"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(
                model="tts-1",
                voice="alloy",
                instructions="Speak in a friendly and conversational tone."
            ),
            vad=silero.VAD.load(),
            turn_detection=turn_detection,
        )
        
        await session.start(
            room=ctx.room,
            agent=FitnessAssistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        
    except Exception as e:
        print(f"Error in entrypoint: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting LiveKit AI Fitness Assistant...")
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
            )
        )
    except Exception as e:
        print(f"Error starting agent: {str(e)}")
        import sys
        sys.exit(1)
