import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
from jsonrpcserver import method, Result, Success, Error
import random
from collections import OrderedDict
import numpy as np
from a3c_model import get_trainer

class RL_Session:
    """Class representing a Reinforcement Learning session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.last_active = self.start_time
        self.access_count = 0  # 访问计数
        self.status = "active"
        self.last_action = None  # 存储最后一个动作，用于奖励回传
        self.last_action_step = 0
        self.last_param = None
        
        # Setup logging
        self.logger = logging.getLogger(f"RL_Session_{session_id}")
        self.logger.info(f"Session {session_id} created.")

    def update_activity(self):
        """Update the last active timestamp and increment access count"""
        self.last_active = datetime.now()
        self.access_count += 1
        # 只在访问计数是10的倍数时记录日志，减少噪音
        if self.access_count % 10 == 0:
            self.logger.info(f"Session {self.session_id} activity updated. Access count: {self.access_count}")
    
    def end_session(self):
        """End the session"""
        self.status = "ended"
        # 通知 A3C 训练器结束会话
        trainer = get_trainer()
        trainer.end_session(self.session_id)
        # self.logger.info(f"Session {self.session_id} ended.")


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
            # self.logger.info(f"Change session ID ignored: new session {new_session_id} already exists")
            return False
        
        # 获取旧session数据
        old_session = self.sessions[old_session_id]
        
        # 创建新session并复制数据
        new_session = RL_Session(new_session_id)
        new_session.start_time = old_session.start_time  # 保持原始开始时间
        new_session.last_active = old_session.last_active  # 保持活跃时间
        new_session.access_count = old_session.access_count  # 保持访问计数
        new_session.status = old_session.status  # 保持状态
        new_session.last_action = old_session.last_action  # 保持最后动作
        new_session.last_action_step = old_session.last_action_step
        new_session.last_param = old_session.last_param
        
        # 插入新session（会自动处理容量限制）
        self.sessions[new_session_id] = new_session
        
        # 结束旧session
        old_session.end_session()
        
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
session_manager = LRUSessionManager(max_capacity=4000)  # 最大1000个session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_server.log'),  # 输出到文件
        logging.StreamHandler()                # 同时输出到控制台
    ]
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
    """Get action from the A3C agent based on the current state"""
    global session_manager
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    # 获取或创建session
    session = session_manager.get_session(session_id)
    if session is None:
        session = session_manager.add_session(session_id)
        # logger.info(f"Auto-created session {session_id}")
    
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
        
        # 减少状态日志输出频率 - 只记录调试级别
        logger.debug(f"Received state: session_id={action_state['session_id'][:8]}..., "
                    f"calls={len(action_state['call_sequence'])} syscalls, "
                    f"exec_time={action_state['exec_time']}ms, "
                    f"errors={action_state['error_count']}")
        
    except Exception as e:
        logger.error(f"Failed to parse state: {e}")
        return Error(-32602, f"Invalid state format: {e}")
    
    # 使用 A3C 模型生成动作
    try:
        trainer = get_trainer()
        action_type, param_probs_dict = trainer.get_action(session_id, action_state)

        # 计算动作参数
        action_param = 0
        if action_type == 0:
            probs0 = param_probs_dict.get("type0", [])
            action_param = int(np.argmax(np.asarray(probs0, dtype=np.float32))) if probs0 else 0
        elif action_type == 2:
            probs2 = param_probs_dict.get("type2", [])
            action_param = int(np.argmax(np.asarray(probs2, dtype=np.float32))) if probs2 else 0
        elif action_type == 3:
            probs3 = param_probs_dict.get("type3", [])
            action_param = int(np.argmax(np.asarray(probs3, dtype=np.float32))) if probs3 else 0
        else:
            action_param = action_type % 3

        # 构造动作
        action = {
            "session_id": session_id,
            "action_type": action_type,
            "action_param": action_param
        }
        
        # 存储动作信息，用于后续的奖励回传
        session.last_action = action_type
        session.last_param = action_param
        session.last_action_step = action_state['call_count']
        
        # 减少动作日志输出频率 - 只记录调试级别
        logger.debug(f"A3C Action provided for session {session_id}: type={action_type}, param={action_param}")
        
    except Exception as e:
        logger.error(f"Error generating action with A3C: {e}")
        # fallback to random action
        x = random.random()
        if x < 0.20:
            action = {"session_id": session_id, "action_type": 0, "action_param": 0}
        elif x < 0.40:
            action = {"session_id": session_id, "action_type": 1, "action_param": 1}
        elif x < 0.60:
            action = {"session_id": session_id, "action_type": 2, "action_param": 2}
        elif x < 0.80:
            action = {"session_id": session_id, "action_type": 3, "action_param": 0}
        else:
            action = {"session_id": session_id, "action_type": 4, "action_param": 1}
        
        session.last_action = action["action_type"]
        session.last_param = action["action_param"]
        session.last_action_step = action_state['call_count']
        logger.debug(f"Fallback random action provided for session {session_id}")
    
    return Success({
        "action": action,
        "session_id": session_id
    })


@method
def submit_reward(session_id: str, reward: float) -> Result:
    """Submit reward to the A3C agent"""
    global session_manager
    
    if not session_id:
        logger.warning("Empty session_id provided")
        return Error(-32602, "Invalid params: session_id is required")
    
    session = session_manager.get_session(session_id)
    if session is None:
        # logger.warning(f"Session {session_id} does not exist")
        return Error(-32603, f"Session {session_id} does not exist")
    
    # 使用 A3C 训练器处理奖励
    try:
        trainer = get_trainer()
        loss = None
        
        if session.last_action is not None:
            action_param = session.last_param if session.last_param is not None else 0
            loss = trainer.submit_reward(session_id, reward, session.last_action, action_param)
            logger.info(
                f"Reward {reward} submitted for session {session_id}, action {session.last_action}, param {action_param}"
            )
            
            if loss is not None:
                logger.info(f"A3C model updated, loss: {loss:.6f}")
        else:
            logger.warning(f"No previous action found for session {session_id}")
            
    except Exception as e:
        logger.error(f"Error submitting reward to A3C: {e}")
        return Error(-32603, f"Error processing reward: {e}")
    
    return Success({
        "success": True,
        "message": f"Reward {reward} submitted successfully",
        "session_id": session_id,
        "loss": loss
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
        # logger.info(f"Session ID changed successfully: {old_session_id} -> {new_session_id}")
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


@method
def get_training_stats() -> Result:
    """Get A3C training statistics"""
    try:
        trainer = get_trainer()
        training_stats = trainer.get_stats()
        logger.info(f"Training stats requested: {training_stats}")
        return Success(training_stats)
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        return Error(-32603, f"Error getting training stats: {e}")


@method
def save_model(model_path: str = None) -> Result:
    """Save the current A3C model"""
    try:
        trainer = get_trainer()
        if model_path is None:
            model_path = f"a3c_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        return Success({
            "success": True,
            "message": f"Model saved to {model_path}",
            "model_path": model_path
        })
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return Error(-32603, f"Error saving model: {e}")


@method
def load_model(model_path: str) -> Result:
    """Load an A3C model"""
    try:
        trainer = get_trainer()
        trainer.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return Success({
            "success": True,
            "message": f"Model loaded from {model_path}",
            "model_path": model_path
        })
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return Error(-32603, f"Error loading model: {e}")

 
@method
def get_exploration_params() -> Result:
    """Get current exploration parameters (epsilon, temperature, action counts)"""
    try:
        trainer = get_trainer()
        agent = trainer.agent
        
        exploration_info = {
            "current_epsilon": agent._current_epsilon(),
            "current_temperature": agent._current_temperature(), 
            "total_action_calls": agent.total_action_calls,
            "epsilon_config": {
                "start": agent.epsilon_start,
                "end": agent.epsilon_end, 
                "decay_steps": agent.epsilon_decay_steps
            },
            "temperature_config": {
                "start": agent.temp_start,
                "end": agent.temp_end,
                "decay_steps": agent.temp_decay_steps
            }
        }
        
        logger.info(f"Exploration params requested: epsilon={exploration_info['current_epsilon']:.4f}, temp={exploration_info['current_temperature']:.4f}")
        return Success(exploration_info)
    except Exception as e:
        logger.error(f"Error getting exploration params: {e}")
        return Error(-32603, f"Error getting exploration params: {e}")



if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer
    from jsonrpcserver import dispatch
    from http.server import BaseHTTPRequestHandler
    import json
    
    parser = argparse.ArgumentParser(description="RL JSON-RPC Server with A3C")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--load-model", type=str, help="Path to load pre-trained model")
    parser.add_argument("--temperature", type=float, help="Initial temperature for exploration (overrides model default)")
    
    args = parser.parse_args()
    
    # 初始化 A3C 训练器
    logger.info("Initializing A3C trainer...")
    trainer = get_trainer()
    
    # 如果指定了模型路径，则加载模型
    if args.load_model:
        try:
            trainer.load_model(args.load_model)
            logger.info(f"Pre-trained model loaded from {args.load_model}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    # 如果指定了温度参数，则设置初始温度
    if args.temperature is not None:
        try:
            trainer.agent.temp_start = args.temperature
            trainer.agent.temp_end = args.temperature  # 固定温度，不衰减
            trainer.agent.temp_decay_steps = 1  # 立即生效
            logger.info(f"Temperature set to {args.temperature}")
        except Exception as e:
            logger.error(f"Failed to set temperature: {e}")
    
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
            # 屏蔽 HTTP 请求日志
            pass
    
    logger.info(f"Starting RL JSON-RPC Server with A3C on {args.host}:{args.port}")
    logger.info("Available methods: ping, init_session, get_action, submit_reward, change_session_id, get_session_stats, get_training_stats, save_model, load_model, get_exploration_params")
    
    # 显示当前探索参数配置
    if args.temperature is not None:
        logger.info(f"Temperature override set to: {args.temperature}")
    try:
        current_epsilon = trainer.agent._current_epsilon()
        current_temp = trainer.agent._current_temperature()
        logger.info(f"Current exploration params - Epsilon: {current_epsilon:.4f}, Temperature: {current_temp:.4f}")
    except Exception as e:
        logger.debug(f"Could not display exploration params: {e}")
    
    try:
        server = HTTPServer((args.host, args.port), RequestHandler)
        logger.info(f"Server listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        # 保存模型
        try:
            final_model_path = f"a3c_model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            trainer.save_model(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        server.shutdown()