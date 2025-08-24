import os
import asyncio
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, silero

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedConfig:
    """Configuration with timeout management"""
    # Models
    STT_MODEL = "whisper-1"
    LLM_MODEL = "gpt-4o-mini"
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_VOICE = "alloy"
    
    # Performance settings
    MAX_AUDIO_BUFFER_SIZE = 512 * 1024
    MAX_RESPONSE_LENGTH = 150
    LLM_TEMPERATURE = 0.3
    
    # TIMEOUT SETTINGS - NEW!
    SESSION_TIMEOUT = 300  # 5 minutes of total session time
    IDLE_TIMEOUT = 60      # 1 minute of no interaction
    WARNING_TIME = 45      # Warn user at 45 seconds of idle
    API_TIMEOUT = 10       # 10 seconds for API calls
    HEALTH_CHECK_INTERVAL = 5  # Check health every 5 seconds
    
    # VAD settings
    VAD_MIN_SILENCE_DURATION = 0.3
    VAD_MIN_SPEECH_DURATION = 0.15

def validate_environment():
    """Validate environment variables efficiently"""
    required = ['OPENAI_API_KEY', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'LIVEKIT_URL']
    missing = [var for var in required if not os.getenv(var) or os.getenv(var).startswith('your_')]
    
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

validate_environment()

class OptimizedFitnessAssistant(Agent):
    """Fitness assistant with timeout and session management"""
    
    def __init__(self) -> None:
        instructions = (
            "You are AndrofitAI, an AI gym coach. Give SHORT, energetic responses (max 30 words). "
            "Instead of counting reps, use motivation: 'Great form!', 'Keep pushing!', 'You got this!'. "
            "Focus on encouragement and form tips. Stay brief and energetic. No counting numbers."
        )
        
        super().__init__(instructions=instructions)

@asynccontextmanager
async def create_optimized_session(ctx: agents.JobContext):
    """Create session with timeout management"""
    session = None
    try:
        session = AgentSession(
            stt=openai.STT(
                model=MemoryOptimizedConfig.STT_MODEL,
                language="en",
            ),
            llm=openai.LLM(
                model=MemoryOptimizedConfig.LLM_MODEL,
                temperature=MemoryOptimizedConfig.LLM_TEMPERATURE,
                max_tokens=MemoryOptimizedConfig.MAX_RESPONSE_LENGTH,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                # Add timeout for LLM calls
                request_timeout=MemoryOptimizedConfig.API_TIMEOUT,
            ),
            tts=openai.TTS(
                model=MemoryOptimizedConfig.TTS_MODEL,
                voice=MemoryOptimizedConfig.TTS_VOICE,
                instructions="Speak with high energy and enthusiasm at a brisk pace.",
                # Add timeout for TTS calls
                request_timeout=MemoryOptimizedConfig.API_TIMEOUT,
            ),
            vad=silero.VAD.load(
                min_silence_duration=MemoryOptimizedConfig.VAD_MIN_SILENCE_DURATION,
                min_speech_duration=MemoryOptimizedConfig.VAD_MIN_SPEECH_DURATION,
                threshold=0.3,
            ),
            preemptive_generation=True,
        )
        
        yield session
        
    except Exception as e:
        logger.error(f"Session initialization error: {e}")
        raise
    finally:
        if session:
            try:
                await session.end()
            except:
                pass

class SessionManager:
    """Manages session timeouts and health monitoring"""
    
    def __init__(self):
        self.session_start_time = time.time()
        self.last_interaction_time = time.time()
        self.warning_sent = False
        self.is_agent_speaking = False
        self.is_user_speaking = False
        
    def reset_interaction_timer(self):
        """Reset the interaction timer"""
        self.last_interaction_time = time.time()
        self.warning_sent = False
        
    def get_idle_time(self):
        """Get current idle time in seconds"""
        return int(time.time() - self.last_interaction_time)
        
    def get_session_duration(self):
        """Get total session duration in seconds"""
        return int(time.time() - self.session_start_time)
        
    def should_send_warning(self):
        """Check if we should send idle warning"""
        idle_time = self.get_idle_time()
        return (idle_time >= MemoryOptimizedConfig.WARNING_TIME and 
                not self.warning_sent and 
                not self.is_agent_speaking and 
                not self.is_user_speaking)
                
    def should_timeout_session(self):
        """Check if session should timeout"""
        idle_time = self.get_idle_time()
        session_time = self.get_session_duration()
        
        # Timeout due to idle time
        idle_timeout = (idle_time >= MemoryOptimizedConfig.IDLE_TIMEOUT and 
                       not self.is_agent_speaking and 
                       not self.is_user_speaking)
        
        # Timeout due to max session time
        session_timeout = session_time >= MemoryOptimizedConfig.SESSION_TIMEOUT
        
        return idle_timeout or session_timeout

async def entrypoint(ctx: agents.JobContext):
    """Optimized entrypoint with comprehensive timeout management"""
    session_manager = SessionManager()
    
    try:
        async with create_optimized_session(ctx) as session:
            agent = OptimizedFitnessAssistant()
            
            # Event handlers for session management
            @session.on("user_started_speaking")
            def on_user_started_speaking():
                session_manager.is_user_speaking = True
                session_manager.reset_interaction_timer()
                logger.info("User started speaking")
                
            @session.on("user_stopped_speaking") 
            def on_user_stopped_speaking():
                session_manager.is_user_speaking = False
                session_manager.reset_interaction_timer()
                logger.info("User stopped speaking")
                
            @session.on("agent_started_speaking")
            def on_agent_started_speaking():
                session_manager.is_agent_speaking = True
                session_manager.reset_interaction_timer()
                logger.info("Agent started speaking")
                
            @session.on("agent_stopped_speaking")
            def on_agent_stopped_speaking():
                session_manager.is_agent_speaking = False
                session_manager.reset_interaction_timer()
                logger.info("Agent stopped speaking")
            
            # Start session
            await session.start(
                room=ctx.room,
                agent=agent,
                room_input_options=RoomInputOptions(auto_subscribe=True),
            )
            
            # Initial greeting
            await session.say(
                "Hey! I'm AndrofitAI. Ready to crush your workout?",
                allow_interruptions=True
            )
            session_manager.reset_interaction_timer()
            
            # Main session loop with timeout monitoring
            while ctx.room.connection_state == "connected":
                try:
                    # Check for idle warning
                    if session_manager.should_send_warning():
                        logger.info("Sending idle warning")
                        session_manager.warning_sent = True
                        await session.say(
                            "Are you still there? Let's keep going!",
                            allow_interruptions=True
                        )
                    
                    # Check for session timeout
                    if session_manager.should_timeout_session():
                        idle_time = session_manager.get_idle_time()
                        session_time = session_manager.get_session_duration()
                        
                        if idle_time >= MemoryOptimizedConfig.IDLE_TIMEOUT:
                            logger.info(f"Session timeout due to {idle_time}s of inactivity")
                            await session.say(
                                "Thanks for the great workout! Stay strong!",
                                allow_interruptions=False
                            )
                        else:
                            logger.info(f"Session timeout after {session_time}s total time")
                            await session.say(
                                "Great session! Time's up. Keep crushing those goals!",
                                allow_interruptions=False
                            )
                        
                        # Graceful disconnect after 2 seconds
                        await asyncio.sleep(2)
                        await ctx.room.disconnect()
                        break
                    
                    # Health check
                    await asyncio.sleep(MemoryOptimizedConfig.HEALTH_CHECK_INTERVAL)
                    
                    # Log session stats periodically
                    if session_manager.get_session_duration() % 30 == 0:
                        logger.info(f"Session active: {session_manager.get_session_duration()}s, "
                                  f"Idle: {session_manager.get_idle_time()}s")
                        
                except Exception as e:
                    logger.error(f"Session monitoring error: {e}")
                    # Try to recover
                    await asyncio.sleep(1)
                    continue
                
    except Exception as e:
        logger.error(f"Session error: {e}")
        # Attempt graceful shutdown
        try:
            if ctx.room.connection_state == "connected":
                await ctx.room.disconnect()
        except:
            pass
        raise

def main():
    """Main function with enhanced error handling"""
    try:
        print("🚀 Starting TimeoutManagedAndrofitAI")
        print("⏰ TIMEOUT MANAGEMENT ENABLED:")
        print(f"  - Max Session Time: {MemoryOptimizedConfig.SESSION_TIMEOUT//60} minutes")
        print(f"  - Idle Timeout: {MemoryOptimizedConfig.IDLE_TIMEOUT} seconds")
        print(f"  - Warning Time: {MemoryOptimizedConfig.WARNING_TIME} seconds")
        print(f"  - API Timeout: {MemoryOptimizedConfig.API_TIMEOUT} seconds")
        print(f"  - Health Checks: Every {MemoryOptimizedConfig.HEALTH_CHECK_INTERVAL} seconds")
        print("⚡ SPEED OPTIMIZATIONS:")
        print(f"  - TTS: {MemoryOptimizedConfig.TTS_MODEL}")
        print(f"  - Preemptive Generation: ENABLED")
        print(f"✅ OpenAI: {'OK' if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_') else 'MISSING'}")
        print(f"✅ LiveKit: {'OK' if os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-') else 'MISSING'}")
        
        worker_options = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        
        agents.cli.run_app(worker_options)
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        return 1
    finally:
        print("🏁 Shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit(main())
