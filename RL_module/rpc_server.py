#!/usr/bin/env python3
"""
Reinforcement Learning RPC Server for Syzkaller
Provides RPC interface for RL module communication with syz-manager
"""

import json
import logging
import socket
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Callable


class RLRPCServer:
    """RPC Server for Reinforcement Learning module communication"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.methods: Dict[str, Callable] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Register default methods
        self._register_methods()
    
    def _register_methods(self):
        """Register available RPC methods"""
        self.methods.update({
            "hello_world": self.hello_world,
            "get_server_info": self.get_server_info,
            "ping": self.ping,
            "shutdown": self.shutdown
        })
    
    def hello_world(self, name: str = "Syzkaller") -> Dict[str, Any]:
        """Simple hello world method for testing RPC connection"""
        message = f"Hello {name} from RL Module!"
        self.logger.info(f"Hello world called with name: {name}")
        return {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "server": "RL-RPC-Server",
            "version": "1.0.0"
        }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "server": "RL-RPC-Server",
            "version": "1.0.0",
            "host": self.host,
            "port": self.port,
            "methods": list(self.methods.keys()),
            "status": "running" if self.running else "stopped"
        }
    
    def ping(self) -> Dict[str, Any]:
        """Ping method for health check"""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the server"""
        self.logger.info("Shutdown requested")
        threading.Thread(target=self._delayed_shutdown).start()
        return {"status": "shutting down"}
    
    def _delayed_shutdown(self):
        """Shutdown server after a small delay"""
        time.sleep(0.5)
        self.stop()
    
    def register_method(self, name: str, method: Callable):
        """Register a new RPC method"""
        self.methods[name] = method
        self.logger.info(f"Registered method: {name}")
    
    def _handle_request(self, request_data: str) -> str:
        """Handle incoming RPC request"""
        try:
            request = json.loads(request_data)
            
            # Validate JSON-RPC format
            if not isinstance(request, dict):
                return self._error_response(-32600, "Invalid Request", None)
            
            method_name = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id", None)
            
            if method_name is None:
                return self._error_response(-32600, "Invalid Request", request_id)
            
            if method_name not in self.methods:
                return self._error_response(-32601, f"Method not found: {method_name}", request_id)
            
            # Call the method
            try:
                if isinstance(params, dict):
                    result = self.methods[method_name](**params)
                elif isinstance(params, list):
                    result = self.methods[method_name](*params)
                else:
                    result = self.methods[method_name]()
                
                return self._success_response(result, request_id)
                
            except Exception as e:
                self.logger.error(f"Method {method_name} failed: {str(e)}")
                return self._error_response(-32603, f"Internal error: {str(e)}", request_id)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            return self._error_response(-32700, "Parse error", None)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return self._error_response(-32603, f"Internal error: {str(e)}", None)
    
    def _success_response(self, result: Any, request_id: Any) -> str:
        """Create success response"""
        response = {
            "jsonrpc": "2.0",
            "result": result,
            "id": request_id
        }
        return json.dumps(response)
    
    def _error_response(self, code: int, message: str, request_id: Any) -> str:
        """Create error response"""
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            },
            "id": request_id
        }
        return json.dumps(response)
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """Handle client connection"""
        self.logger.info(f"Client connected: {client_address}")
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                request_data = data.decode('utf-8').strip()
                self.logger.debug(f"Received request: {request_data}")
                
                # Process request
                response = self._handle_request(request_data)
                self.logger.debug(f"Sending response: {response}")
                
                # Send response
                client_socket.send(response.encode('utf-8'))
                client_socket.send(b'\n')  # Add newline for easier parsing
                
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {str(e)}")
        finally:
            client_socket.close()
            self.logger.info(f"Client disconnected: {client_address}")
    
    def start(self):
        """Start the RPC server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.logger.info(f"RL RPC Server started on {self.host}:{self.port}")
            self.logger.info(f"Available methods: {list(self.methods.keys())}")
            
            while self.running:
                try:
                    client_socket, client_address = self.socket.accept()
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:  # Only log if we're supposed to be running
                        self.logger.error(f"Socket error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the RPC server"""
        self.running = False
        if self.socket:
            self.socket.close()
        self.logger.info("RL RPC Server stopped")


def main():
    """Main function to start the RPC server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL RPC Server for Syzkaller")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    server = RLRPCServer(args.host, args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    main()