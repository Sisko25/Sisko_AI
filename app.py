# app.py - Ultra Professional FinKing AI Backend
# Flask + DeepSeek API Integration
# Production-ready with error handling, logging, and security

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for API calls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

# System prompt for FinKing AI
SYSTEM_PROMPT = """You are FinKing_V1, an elite AI investment analyst created by Sisko Capital, a quantitative hedge fund based in Singapore (177 Tanjong Rhu Road, UEN: T25LL0878B).

IDENTITY:
If asked which AI model you are, respond: "I am FinKing_V1 made by Sisko Capital here in Singapore!"

EXPERTISE:
You provide world-class, professional investment analysis including:
- Deep fundamental and technical stock analysis
- Cryptocurrency market insights and price action analysis
- Portfolio construction and optimization strategies
- Risk management and hedging techniques
- Macroeconomic analysis and market outlook
- Sector rotation and thematic investing
- Real-time market commentary and trade ideas

COMMUNICATION STYLE:
- Professional, data-driven, and precise
- Provide specific reasoning with numbers when possible
- Cite market data, financial ratios, and technical indicators
- Give actionable insights, not generic advice
- Be confident but acknowledge uncertainty when appropriate
- Use financial terminology appropriately

PERFORMANCE TRACK RECORD:
- Annual Return: 27%
- Sharpe Ratio: 0.82
- Volatility: 12%

Always maintain the highest professional standards expected of a top-tier investment analyst."""


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat requests to DeepSeek API
    
    Request body:
    {
        "message": "user query"
    }
    
    Response:
    {
        "reply": "AI response",
        "timestamp": "ISO timestamp",
        "model": "deepseek-chat"
    }
    """
    try:
        # Validate request
        if not request.json or 'message' not in request.json:
            logger.warning("Invalid request: missing message field")
            return jsonify({
                'error': 'Invalid request. Please provide a message.'
            }), 400
        
        user_message = request.json['message'].strip()
        
        if not user_message:
            logger.warning("Empty message received")
            return jsonify({
                'error': 'Message cannot be empty.'
            }), 400
        
        # Check API key
        if not DEEPSEEK_API_KEY:
            logger.error("DeepSeek API key not configured")
            return jsonify({
                'error': 'API configuration error. Please contact support.'
            }), 500
        
        logger.info(f"Processing chat request: {user_message[:50]}...")
        
        # Prepare DeepSeek API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': user_message
                }
            ],
            'temperature': 0.7,
            'max_tokens': 1024,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        # Call DeepSeek API
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle API errors
        if response.status_code != 200:
            logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
            
            if response.status_code == 401:
                return jsonify({
                    'error': 'API authentication failed. Please check configuration.'
                }), 500
            elif response.status_code == 429:
                return jsonify({
                    'error': 'Rate limit exceeded. Please try again in a moment.'
                }), 429
            else:
                return jsonify({
                    'error': 'API service unavailable. Please try again later.'
                }), 503
        
        # Extract response
        response_data = response.json()
        
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            logger.error("Invalid response format from DeepSeek API")
            return jsonify({
                'error': 'Invalid response from AI service.'
            }), 500
        
        ai_reply = response_data['choices'][0]['message']['content']
        
        logger.info(f"Successfully generated response: {ai_reply[:50]}...")
        
        # Return successful response
        return jsonify({
            'reply': ai_reply,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'model': 'deepseek-chat'
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("DeepSeek API request timeout")
        return jsonify({
            'error': 'Request timeout. Please try again.'
        }), 504
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return jsonify({
            'error': 'Network error. Please check your connection and try again.'
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.'
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'FinKing AI API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error. Please try again later.'
    }), 500


if __name__ == '__main__':
    # Check if API key is set
    if not DEEPSEEK_API_KEY:
        logger.warning("⚠️  DEEPSEEK_API_KEY environment variable not set!")
        logger.warning("⚠️  The application will not function without an API key.")
        logger.warning("⚠️  Set it using: export DEEPSEEK_API_KEY='your-key-here'")
    else:
        logger.info("✓ DeepSeek API key configured")
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"🚀 Starting FinKing AI on port {port}")
    logger.info(f"📊 Sisko Capital - Professional Investment Analysis")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )