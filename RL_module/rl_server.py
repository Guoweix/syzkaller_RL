
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
from jsonrpcserver import method, serve, Result, Success, Error
import random
from collections import OrderedDict

class RL_Session:
    """Class representing a Reinforcement Learning session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.last_active = self.start_time
        self.access_count = 0  # 访问计数
        self.status = "active"
        
        # Setup logging
        self.logger = logging.getLogger(f"RL_Session_{session_id}")
        self.logger.info(f"Session {session_id} created.")

    def update_activity(self):
        """Update the last active timestamp and increment access count"""
        self.last_active = datetime.now()
        self.access_count += 1
        self.logger.info(f"Session {self.session_id} activity updated. Access count: {self.access_count}")
    
    def end_session(self):
        """End the session"""
        self.status = "ended"
        self.logger.info(f"Session {self.session_id} ended.")


class LRUSessionManager:
    """LRU Session Manager with maximum capacity"""
    
    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.sessions = OrderedDict()  # 使用OrderedDict维护访问顺序
        self.logger = logging.getLogger("LRUSessionManager")
        
    def get_session(self, session_id: str) -> Optional[RL_Session]:
        """获取session，同时更新访问顺序"""
        if session_id in self.sessions:
            # 移动到最后（最近访问）
            session = self.sessions.pop(session_id)
            self.sessions[session_id] = session
            session.update_activity()
            return session
        return None
    
    def add_session(self, session_id: str) -> RL_Session:
        """添加新session，如果超过容量则删除最少访问的"""
        # 如果已存在，直接返回
        if session_id in self.sessions:
            return self.get_session(session_id)
        
        # 检查是否需要删除最少使用的session
        if len(self.sessions) >= self.max_capacity:
            # 删除最老的（最少访问的）session
            oldest_id, oldest_session = self.sessions.popitem(last=False)
            oldest_session.end_session()
            self.logger.info(f"Removed LRU session {oldest_id} due to capacity limit ({self.max_capacity})")
        
        # 添加新session
        new_session = RL_Session(session_id)
        self.sessions[session_id] = new_session
        self.logger.info(f"Added new session {session_id}. Total sessions: {len(self.sessions)}")
        return new_session
    
    def remove_session(self, session_id: str) -> bool:
        """删除指定session"""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            session.end_session()
            self.logger.info(f"Removed session {session_id}. Total sessions: {len(self.sessions)}")
            return True
        return False
    
    def session_exists(self, session_id: str) -> bool:
        """检查session是否存在"""
        return session_id in self.sessions
    
    def change_session_id(self, old_session_id: str, new_session_id: str) -> bool:
        """将旧session的数据复制到新session ID，旧session由LRU自然清理"""
        # 检查旧session是否存在
        if old_session_id not in self.sessions:
            self.logger.info(f"Change session ID ignored: old session {old_session_id} does not exist")
            return False
        
        # 检查新session ID是否已存在
        if new_session_id in self.sessions:
            self.logger.info(f"Change session ID ignored: new session {new_session_id} already exists")
            return False
        
        # 获取旧session数据
        old_session = self.sessions[old_session_id]
        
        # 创建新session并复制数据
        new_session = RL_Session(new_session_id)
        new_session.start_time = old_session.start_time  # 保持原始开始时间
        new_session.last_active = old_session.last_active  # 保持活跃时间
        new_session.access_count = old_session.access_count  # 保持访问计数
        new_session.status = old_session.status  # 保持状态
        
        # 插入新session（会自动处理容量限制）
        self.sessions[new_session_id] = new_session
        
        self.logger.info(f"Session ID changed: {old_session_id} -> {new_session_id}, old session will be cleaned by LRU")
        return True
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_sessions": len(self.sessions),
            "max_capacity": self.max_capacity,
            "capacity_usage": f"{len(self.sessions)}/{self.max_capacity}",
            "usage_percentage": round((len(self.sessions) / self.max_capacity) * 100, 2)
        }


# Global session manager with LRU capability
session_manager = LRUSessionManager(max_capacity=1000)  # 最大1000个session

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
    global session_manager
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    if session_manager.session_exists(session_id):
        logger.warning(f"Session {session_id} already exists")
        return Error(-32603, f"Session {session_id} already exists")
    
    session_manager.add_session(session_id)
    stats = session_manager.get_stats()
    logger.info(f"Session {session_id} initialized. Stats: {stats}")
    return Success({
        "success": True,
        "message": f"Session {session_id} initialized successfully",
        "session_id": session_id,
        "stats": stats
    })


@method
def get_action(session_id: str, state: Dict) -> Result:
    """Get action from the RL agent based on the current state"""
    global session_manager
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    # 获取或创建session
    session = session_manager.get_session(session_id)
    if session is None:
        session = session_manager.add_session(session_id)
        logger.info(f"Auto-created session {session_id}")
    
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

    # action = {"session_id": session_id, "action_type": 0, "action_param": 1}
    x=random.random()

    if x<0.10:
          action = {"session_id": session_id, "action_type": 0, "action_param": 0}
    elif x<0.20:
            action = {"session_id": session_id, "action_type": 1, "action_param": 1}
    elif x<0.30:
            action = {"session_id": session_id, "action_type": 2, "action_param": 2}
    elif x<0.40:
            action = {"session_id": session_id, "action_type": 3, "action_param": 0}
    elif x<0.50:
            action = {"session_id": session_id, "action_type": 4, "action_param": 1}
    else:   
            action = {"session_id": session_id, "action_type": 5, "action_param": 2}
    
    
    # session activity 已经在get_session中更新了
    logger.info(f"Action provided for session {session_id}")
    return Success({
        "action": action,
        "session_id": session_id
    })


@method
def submit_reward(session_id: str, reward: float) -> Result:
    """Submit reward to the RL agent"""
    global session_manager
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    session = session_manager.get_session(session_id)
    if session is None:
        logger.warning(f"Session {session_id} does not exist")
        return Error(-32603, f"Session {session_id} does not exist")
    
    # Placeholder for reward processing logic
    logger.info(f"Reward {reward} submitted for session {session_id}")
    
    return Success({
        "success": True,
        "message": f"Reward {reward} submitted successfully",
        "session_id": session_id
    })


@method
def change_session_id(old_session_id: str, new_session_id: str) -> Result:
    """Change session ID by copying old session data to new ID"""
    global session_manager
    
    if not old_session_id or not new_session_id:
        logger.warning("Empty session ID(s) provided for change_session_id")
        return Error(-32602, "Invalid params: both old_session_id and new_session_id are required")
    
    success = session_manager.change_session_id(old_session_id, new_session_id)
    
    if success:
        logger.info(f"Session ID changed successfully: {old_session_id} -> {new_session_id}")
        return Success({
            "success": True,
            "message": f"Session ID changed from {old_session_id} to {new_session_id}",
            "old_session_id": old_session_id,
            "new_session_id": new_session_id
        })
    else:
        logger.info(f"Session ID change ignored: {old_session_id} -> {new_session_id}")
        return Success({
            "success": False,
            "message": "Session ID change ignored (session not found or target exists)",
            "old_session_id": old_session_id,
            "new_session_id": new_session_id
        })


@method
def get_session_stats() -> Result:
    """Get session manager statistics"""
    global session_manager
    
    stats = session_manager.get_stats()
    logger.info(f"Session stats requested: {stats}")
    return Success(stats)

    

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
    logger.info("Available methods: ping, init_session, get_action, submit_reward, change_session_id, get_session_stats")
    
    try:
        server = HTTPServer((args.host, args.port), RequestHandler)
        logger.info(f"Server listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        server.shutdown()