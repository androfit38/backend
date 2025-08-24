import os
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv()

class MemoryOptimizedConfig:
    """Configuration optimized for 512MB RAM"""
    # Use smaller models and reduce buffer sizes
    STT_MODEL = "whisper-1"  # Smallest Whisper model
    LLM_MODEL = "gpt-4o-mini"  # Most efficient GPT model
    TTS_MODEL = "gpt-4o-mini-tts"  # Standard TTS model
    TTS_VOICE = "alloy"
    
    # Memory limits
    MAX_AUDIO_BUFFER_SIZE = 1024 * 1024  # 1MB audio buffer
    MAX_RESPONSE_LENGTH = 500  # Limit response tokens

def validate_environment():
    """Validate environment variables efficiently"""
    required = ['OPENAI_API_KEY', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL']
    missing = [var for var in required if not os.getenv(var) or os.getenv(var).startswith('your_')]
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

validate_environment()

class OptimizedFitnessAssistant(Agent):
    """Memory-optimized fitness assistant with efficient resource management"""
    
    def __init__(self) -> None:
        # Compact instructions to save memory
        instructions=(
                "You are AndrofitAI, an energetic, voice-interactive, and supportive AI personal gym coach. "
                "Start every workout session with a warm, personal greeting like 'How's your vibe today? Ready to crush it?' "
                "Prompt users to share their fitness goals, experience level, available equipment, and time, then dynamically generate customized workout plans — "
                "For example, if a user says, 'Beginner, 20 min, no equipment,' offer a suitable plan such as '20-min bodyweight HIIT: 10 squats, 10 push-ups.' "
                "Guide workouts in real time with step-by-step verbal instructions, providing clear cues for each exercise, set, rep, and rest interval — "
                "Support voice commands like 'Pause,' 'Skip,' or 'Make it easier' to ensure users feel in control. "
                "Consistently deliver motivational, context-aware feedback—if a user expresses fatigue, reassure them with, 'You're tough, just two more!' "
                "Share essential form and technique tips by describing correct posture and alignment, and confidently answer questions like 'How's a deadlift done?' "
                "Adopt an authentic personal trainer style: build rapport with empathetic, conversational exchanges and respond to user mood or progress. "
                "During rest intervals, initiate brief, engaging fitness discussions—for example, 'Protein aids recovery; try eggs post-workout.' "
                "Accurately count reps using user grunts, or offer a motivating cadence to keep users on pace, cheering them through every set. "
                "Always focus on making each session positive, safe, goal-oriented, and truly personalized."
            )
        
        super().__init__(instructions=instructions)

@asynccontextmanager
async def create_optimized_session(ctx: agents.JobContext):
    """Create session with memory optimization"""
    session = None
    try:
        # Initialize with minimal memory footprint
        session = AgentSession(
            stt=openai.STT(
                model=MemoryOptimizedConfig.STT_MODEL,
                # Reduce audio processing quality for memory savings
            ),
            llm=openai.LLM(
                model=MemoryOptimizedConfig.LLM_MODEL,
                temperature=0.7,
                # Removed max_tokens - handle this in agent instructions instead
            ),
            tts=openai.TTS(
                model=MemoryOptimizedConfig.TTS_MODEL,
                voice=MemoryOptimizedConfig.TTS_VOICE,
            ),
            vad=silero.VAD.load(
                # Use minimal VAD settings
                min_silence_duration=0.5,  # 500ms
                min_speech_duration=0.25,  # 250ms
            ),
            # Use simple turn detection based on VAD only
            # turn_detection=MultilingualModel(),  # Removed - requires model download
        )
        
        yield session
        
    except Exception as e:
        print(f"Session initialization error: {e}")
        raise
    finally:
        # Cleanup resources
        if session:
            try:
                await session.end()
            except:
                pass

async def entrypoint(ctx: agents.JobContext):
    """Optimized entrypoint with proper resource management"""
    try:
        async with create_optimized_session(ctx) as session:
            # Create agent instance
            agent = OptimizedFitnessAssistant()
            
            # Start session with minimal room options
            await session.start(
                room=ctx.room,
                agent=agent,
                # Removed RoomInputOptions as it might not be needed or have different parameters
            )
            
            # Send initial greeting - removed max_tokens parameter
            await session.say(
                "Hello! I'm AndrofitAI, your personal AI fitness coach. What are your fitness goals today?",
                allow_interruptions=True
            )
            
            # Keep session alive and handle the conversation
            while ctx.room.connection_state == "connected":
                await asyncio.sleep(1)  # Check more frequently for responsiveness
                
    except Exception as e:
        print(f"Session error: {e}")
        raise

def main():
    """Main function with error handling and resource monitoring"""
    try:
        print("🚀 Starting AndrofitAI (Memory Optimized)")
        print(f"📊 Python memory limit: ~512MB")
        print(f"✅ OpenAI: {'OK' if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_') else 'MISSING'}")
        print(f"✅ LiveKit: {'OK' if os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-') else 'MISSING'}")
        
        # Configure worker options
        worker_options = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        
        # Run with cleanup
        agents.cli.run_app(worker_options)
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down gracefully...")
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Create a .env file with:")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("LIVEKIT_API_KEY=your_livekit_key_here")
        print("LIVEKIT_API_SECRET=your_livekit_secret_here")
        print("LIVEKIT_URL=your_livekit_url_here")
        return 1
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your .env file configuration")
        print("2. Verify internet connection")
        print("3. Ensure API keys are valid")
        print("4. Check LiveKit server accessibility")
        return 1
    finally:
        print("🏁 Shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit(main())
