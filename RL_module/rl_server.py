
import logging
import asyncio
from datetime import datetime
from typing import Dict
from jsonrpcserver import method, serve, Result, Success, Error

class RL_Session:
    """Class representing a Reinforcement Learning session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.last_active = self.start_time
        self.status = "active"
        
        # Setup logging
        self.logger = logging.getLogger(f"RL_Session_{session_id}")
        self.logger.info(f"Session {session_id} created.")

    def update_activity(self):
        """Update the last active timestamp"""
        self.last_active = datetime.now()
        self.logger.info(f"Session {self.session_id} activity updated.")
    
    def end_session(self):
        """End the session"""
        self.status = "ended"
        self.logger.info(f"Session {self.session_id} ended.")


# Global session storage
sessions: Dict[str, RL_Session] = {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RL_JSON_RPC_Server")


@method
def ping() -> Result:
    """Ping the server for health check"""
    logger.info("Ping received")
    return Success({
        "status": "pong",
        "timestamp": datetime.now().isoformat()
    })


@method
def init_session(session_id: str) -> Result:
    """Initialize a new RL session"""
    global sessions
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    if session_id in sessions:
        logger.warning(f"Session {session_id} already exists")
        return Error(-32603, f"Session {session_id} already exists")
    
    sessions[session_id] = RL_Session(session_id)
    logger.info(f"Session {session_id} initialized")
    return Success({
        "success": True,
        "message": f"Session {session_id} initialized successfully",
        "session_id": session_id
    })


@method
def end_session(session_id: str) -> Result:
    """End an existing RL session"""
    global sessions
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    if session_id not in sessions:
        logger.warning(f"Session {session_id} does not exist")
        return Error(-32603, f"Session {session_id} does not exist")
    
    sessions[session_id].end_session()
    del sessions[session_id]
    logger.info(f"Session {session_id} ended")
    return Success({
        "success": True,
        "message": f"Session {session_id} ended successfully",
        "session_id": session_id
    })

@method
def get_action(session_id: str, state: Dict) -> Result:
    """Get action from the RL agent based on the current state"""
    global sessions
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    if session_id not in sessions:
        logger.warning(f"Session {session_id} does not exist")
        return Error(-32603, f"Session {session_id} does not exist")
    
    # 将Go客户端传来的state转换成Python可用的结构体
    try:
        # 提取ActionState字段
        action_state = {
            'session_id': state.get('session_id', ''),
            'call_sequence': state.get('call_sequence', []),  # 系统调用ID序列
            'call_count': state.get('call_count', 0),
            'exec_time': state.get('exec_time', 0),  # 执行时间(毫秒)
            'error_count': state.get('error_count', 0)
        }
        
        logger.info(f"Received state: session_id={action_state['session_id'][:8]}..., "
                   f"calls={action_state['call_sequence']}, "
                   f"exec_time={action_state['exec_time']}ms, "
                   f"errors={action_state['error_count']}")
        
    except Exception as e:
        logger.error(f"Failed to parse state: {e}")
        return Error(-32602, f"Invalid state format: {e}")
    
    # 基于转换后的状态生成动作
    action = {"session_id": session_id, "action_type": 0, "action_param": 1}
    
    sessions[session_id].update_activity()
    logger.info(f"Action provided for session {session_id}")
    return Success({
        "action": action,
        "session_id": session_id
    })


@method
def submit_reward(session_id: str, reward: float) -> Result:
    """Submit reward to the RL agent"""
    global sessions
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    if session_id not in sessions:
        logger.warning(f"Session {session_id} does not exist")
        return Error(-32603, f"Session {session_id} does not exist")
    
    # Placeholder for reward processing logic
    logger.info(f"Reward {reward} submitted for session {session_id}")
    
    sessions[session_id].update_activity()
    return Success({
        "success": True,
        "message": f"Reward {reward} submitted successfully",
        "session_id": session_id
    })

    

if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer
    from jsonrpcserver import dispatch
    from http.server import BaseHTTPRequestHandler
    import json
    
    parser = argparse.ArgumentParser(description="RL JSON-RPC Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    
    args = parser.parse_args()
    
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            data = self.rfile.read(content_length).decode()
            
            response = dispatch(data)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
        
        def log_message(self, format, *args):
            logger.info(f"{self.address_string()} - {format % args}")
    
    logger.info(f"Starting RL JSON-RPC Server on {args.host}:{args.port}")
    logger.info("Available methods: ping, init_session, end_session")
    
    try:
        server = HTTPServer((args.host, args.port), RequestHandler)
        logger.info(f"Server listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        server.shutdown()