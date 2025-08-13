from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import os
import asyncio
import signal

# Load environment variables from .env file
load_dotenv()

class FitnessAssistant(Agent):
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    """
    def __init__(self) -> None:  # Fixed: __init__ instead of init
        super().__init__(  # Fixed: __init__ instead of init
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
        session = AgentSession(
            stt=openai.STT(
                model="whisper-1",
            ),
            llm=openai.LLM(
                model="gpt-4o-mini"  # Fixed: corrected model name
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="alloy",
                instructions="Speak in a friendly and conversational tone."
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        
        # Start the session with the FitnessAssistant agent
        await session.start(
            room=ctx.room,
            agent=FitnessAssistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        # Greet the user and offer assistance
        await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
        
    except Exception as e:
        print(f"Error in entrypoint: {str(e)}")
        raise

async def main():
    """Main function to run the agent with proper error handling"""
    try:
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        
        # Run the agent app
        await agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
            )
        )
    except Exception as e:
        print(f"Error starting agent: {str(e)}")
        raise

async def shutdown():
    """Graceful shutdown handler"""
    print("Shutting down gracefully...")
    # Add cleanup code here if needed
    
if __name__ == "__main__":  # Fixed: __name__ instead of name
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Application interrupted")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import sys
        sys.exit(1)
