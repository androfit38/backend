import os
import asyncio
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import threading
import signal
import sys

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app for health checks and web service requirements
app = Flask(__name__)

@app.route('/')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "service": "AndrofitAI Agent",
        "message": "LiveKit agent is running"
    })

@app.route('/health')
def health():
    """Additional health endpoint"""
    return jsonify({"status": "ok"})

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
            "\n\nPlease set environment variables in Render dashboard.\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

class FitnessAssistant(Agent):
    """
    AndrofitAI: An energetic, voice-interactive, and supportive AI personal gym coach.
    Guides users through personalized workout sessions with motivational feedback and real-time instructions.
    """
    def __init__(self) -> None:
        super().__init__(
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
        )

async def entrypoint(ctx: agents.JobContext):
    """Main agent entrypoint"""
    try:
        logger.info("Initializing agent session...")
        # Initialize session with proper error handling
        session = AgentSession(
            stt=openai.STT(
                model="whisper-1",
            ),
            llm=openai.LLM(
                model="gpt-4o-mini"
            ),
            tts=openai.TTS(
                model="tts-1",
                voice="alloy",
                instructions="Speak in a friendly and conversational tone."
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )
        logger.info("Agent session initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent session: {str(e)}")
        raise

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

def run_agent():
    """Run the LiveKit agent"""
    try:
        logger.info("Starting AndrofitAI agent...")
        logger.info(f"OpenAI API Key configured: {'✓' if os.getenv('OPENAI_API_KEY') and not os.getenv('OPENAI_API_KEY').startswith('your_') else '✗'}")
        logger.info(f"LiveKit configured: {'✓' if os.getenv('LIVEKIT_URL') and not os.getenv('LIVEKIT_URL').startswith('wss://your-') else '✗'}")
        
        # Run the agent app
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                ws_url=os.getenv("LIVEKIT_URL"),
                api_key=os.getenv("LIVEKIT_API_KEY"),
                api_secret=os.getenv("LIVEKIT_API_SECRET"),
            )
        )
    except ValueError as e:
        logger.error(f"Configuration Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        logger.error("Troubleshooting tips:")
        logger.error("1. Ensure environment variables are properly configured")
        logger.error("2. Check your internet connection")
        logger.error("3. Verify your API keys are valid")
        logger.error("4. Make sure LiveKit server is accessible")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, exiting...")
    sys.exit(0)

def main():
    """Main function to run both Flask and LiveKit agent"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Validate environment variables
        validate_environment()
        
        # Start the agent in a separate thread
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        
        # Get port from environment variable (required by Render)
        port = int(os.environ.get("PORT", 10000))
        
        # Run Flask app
        logger.info(f"Starting web server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
