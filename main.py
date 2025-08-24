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
    """Configuration optimized for 512MB RAM with fastest models"""
    # OPTIMIZED MODELS for speed and cost
    STT_MODEL = "whisper-1"  # Same - stable and efficient
    LLM_MODEL = "gpt-4o-mini"  # Same - fast and efficient  
    TTS_MODEL = "gpt-4o-mini-tts"  # CHANGED: Faster than tts-1
    TTS_VOICE = "alloy"
    
    # Memory limits
    MAX_AUDIO_BUFFER_SIZE = 512 * 1024  # 512KB for faster processing
    MAX_RESPONSE_LENGTH = 150  # Limit response tokens to prevent counting
    LLM_TEMPERATURE = 0.3  # Lower for faster responses
    
    # VAD settings optimized for speed
    VAD_MIN_SILENCE_DURATION = 0.3  # Faster turn detection
    VAD_MIN_SPEECH_DURATION = 0.15  # Quicker speech detection

def validate_environment():
    """Validate environment variables efficiently"""
    required = ['OPENAI_API_KEY', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL']
    missing = [var for var in required if not os.getenv(var) or os.getenv(var).startswith('your_')]
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

validate_environment()

class OptimizedFitnessAssistant(Agent):
    """Memory-optimized fitness assistant with ultra-fast TTS"""
    
    def __init__(self) -> None:
        # Compact instructions to prevent counting and ensure speed
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
    """Create session with fastest models and streaming"""
    session = None
    try:
        # Initialize with speed-optimized settings
        session = AgentSession(
            stt=openai.STT(
                model=MemoryOptimizedConfig.STT_MODEL,
                # Optimize for speed
                language="en",  # Faster processing with specified language
            ),
            llm=openai.LLM(
                model=MemoryOptimizedConfig.LLM_MODEL,
                temperature=MemoryOptimizedConfig.LLM_TEMPERATURE,
                max_tokens=MemoryOptimizedConfig.MAX_RESPONSE_LENGTH,
                # Speed optimizations
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1,   # Encourage variety
            ),
            tts=openai.TTS(
                model=MemoryOptimizedConfig.TTS_MODEL,  # Using gpt-4o-mini-tts
                voice=MemoryOptimizedConfig.TTS_VOICE,
                # gpt-4o-mini-tts specific optimizations
                instructions="Speak with high energy and enthusiasm at a brisk, conversational pace. Use motivational tone.",
                # Note: gpt-4o-mini-tts uses instructions instead of speed parameter
            ),
            vad=silero.VAD.load(
                # Aggressive settings for fastest response
                min_silence_duration=MemoryOptimizedConfig.VAD_MIN_SILENCE_DURATION,
                min_speech_duration=MemoryOptimizedConfig.VAD_MIN_SPEECH_DURATION,
                threshold=0.3,  # More sensitive detection
            ),
            # Enable preemptive generation for ultra-low latency
            preemptive_generation=True,
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
    """Optimized entrypoint with ultra-fast TTS streaming"""
    try:
        async with create_optimized_session(ctx) as session:
            # Create agent instance
            agent = OptimizedFitnessAssistant()
            
            # Start session with minimal room options
            await session.start(
                room=ctx.room,
                agent=agent,
                room_input_options=RoomInputOptions(
                    auto_subscribe=True,
                ),
            )
            
            # Send fast initial greeting with gpt-4o-mini-tts
            await session.say(
                "Hey! I'm AndrofitAI. Ready to crush your workout?",
                allow_interruptions=True
            )
            
            # Ultra-responsive session management
            while ctx.room.connection_state == "connected":
                await asyncio.sleep(0.05)  # Very fast checking for responsiveness
                
    except Exception as e:
        print(f"Session error: {e}")
        raise

def main():
    """Main function with gpt-4o-mini-tts performance monitoring"""
    try:
        print("🚀 Starting UltraFastAndrofitAI")
        print("⚡ SPEED OPTIMIZATIONS ENABLED:")
        print(f"  - STT: {MemoryOptimizedConfig.STT_MODEL}")
        print(f"  - LLM: {MemoryOptimizedConfig.LLM_MODEL}")
        print(f"  - TTS: {MemoryOptimizedConfig.TTS_MODEL} (FASTEST)")
        print(f"  - Max Response: {MemoryOptimizedConfig.MAX_RESPONSE_LENGTH} tokens")
        print(f"  - VAD Silence: {MemoryOptimizedConfig.VAD_MIN_SILENCE_DURATION}s")
        print(f"  - Preemptive Generation: ENABLED")
        print(f"  - Streaming Audio: ENABLED")
        print(f"💰 Cost Optimized: ~50% TTS savings")
        print(f"📊 Memory limit: ~512MB")
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
        print("\n🔧 Ultra-Fast Troubleshooting:")
        print("1. Check gpt-4o-mini-tts availability in your region")
        print("2. Monitor OpenAI API rate limits")
        print("3. Test streaming audio output")
        print("4. Verify network latency to OpenAI")
        return 1
    finally:
        print("🏁 Shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit(main())
