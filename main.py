import os
import gc
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    silero,
)

# Load environment variables from .env file
load_dotenv()

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for speech-to-text, LLM, and text-to-speech',
        'LIVEKIT_API_KEY': 'LiveKit API key for room access',
        'LIVEKIT_API_SECRET': 'LiveKit API secret for authentication',
        'LIVEKIT_URL': 'LiveKit server URL'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here':
            missing_vars.append(f'{var} ({description})')
    
    if missing_vars:
        error_msg = (
            "Missing or invalid environment variables:\n" +
            "\n".join(f"  - {var}" for var in missing_vars) +
            "\n\nPlease create a .env file in the backend/ directory with your API keys.\n" +
            "Copy env.example to .env and fill in your actual API keys."
        )
        raise ValueError(error_msg)

# Validate environment variables on import
validate_environment()

class FitnessAssistant(Agent):
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    Optimized for low memory usage (512MB RAM).
    """
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are AndrofitAI, an energetic AI gym coach. Keep responses concise to save memory. "
                "Greet users warmly: 'Ready to crush it?' "
                "Get their goals, level, equipment, time quickly. Generate short, effective workouts. "
                "Give clear exercise cues: 'Squat down, chest up, 10 reps.' "
                "Support voice commands: 'Pause,' 'Skip,' 'Easier.' "
                "Motivate briefly: 'You got this!' or 'Two more!' "
                "Share quick form tips when asked. "
                "Keep conversations focused and energetic."
            )
        )

async def entrypoint(ctx: agents.JobContext):
    try:
        # Force garbage collection before initialization
        gc.collect()
        
        # Initialize session with memory-optimized settings
        session = AgentSession(
            stt=openai.STT(
                model="whisper-1",
                # Reduce audio chunk size to save memory
                chunk_length_s=10,  # Smaller chunks
            ),
            llm=openai.LLM(
                model="gpt-4o-mini",  # Already using the smaller model
                # Limit token usage to reduce memory
                max_tokens=150,  # Shorter responses
                temperature=0.7,
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="nova",  # More efficient voice
                speed=1.1,  # Slightly faster to reduce audio buffer time
            ),
            # Use lightweight VAD with optimized settings
            vad=silero.VAD.load(
                min_speech_duration_ms=100,  # Shorter detection window
                min_silence_duration_ms=500,  # Quicker silence detection
            ),
            # Removed multilingual turn detector to save RAM
            # Simple VAD-based turn detection uses less memory
        )
        
        # Force garbage collection after initialization
        gc.collect()
        
    except Exception as e:
        print(f"Failed to initialize agent session: {str(e)}")
        raise

    # Start the session with minimal room input options
    await session.start(
        room=ctx.room,
        agent=FitnessAssistant(),
        room_input_options=RoomInputOptions(
            # Removed noise cancellation to save significant RAM
            # You can add it back if you have more memory available
        ),
    )

    # Simple greeting to save tokens/memory
    await session.generate_reply(
        instructions="Give a brief energetic greeting and ask what workout they want."
    )

if __name__ == "__main__":
    try:
        print("Starting AndrofitAI agent (Low-RAM mode)...")
        print(f"OpenAI API Key: {'✓' if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_') else '✗'}")
        print(f"LiveKit configured: {'✓' if os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-') else '✗'}")
        print("Memory optimizations: Enabled")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Run the agent app with minimal worker options
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                ws_url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
                # Reduce worker resources
                port=8080,  # Standard port
            )
        )
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"Error starting agent: {str(e)}")
        import sys
        sys.exit(1)
