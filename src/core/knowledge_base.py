import json
import os
from typing import Dict, Any, List
import threading
import logging
from datetime import datetime, timedelta
import shutil
import schedule
import time
from fuzzywuzzy import fuzz
import redis

class KnowledgeBase:
    def __init__(self, file_path: str = "knowledge_base.json", backup_dir: str = "backups"):
        self.file_path = file_path
        self.backup_dir = backup_dir
        self.knowledge: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.load_knowledge()
        
        logging.basicConfig(filename='knowledge_base.log', level=logging.INFO)
        
        # Redis connection for distributed storage
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Start backup scheduler
        schedule.every().day.at("00:00").do(self.create_backup)
        threading.Thread(target=self.run_scheduler, daemon=True).start()

    def load_knowledge(self):
        if os.path.exists(self.file_path):
            with self.lock:
                with open(self.file_path, 'r') as f:
                    self.knowledge = json.load(f)
        else:
            self.knowledge = {}

    def save_knowledge(self):
        with self.lock:
            with open(self.file_path, 'w') as f:
                json.dump(self.knowledge, f, indent=2)

    def get(self, key: str) -> Any:
        with self.lock:
            data = self.knowledge.get(key)
            if data and self.is_data_expired(data):
                del self.knowledge[key]
                self.save_knowledge()
                return None
            return data

    def set(self, key: str, value: Any):
        with self.lock:
            if not self.validate_data(value):
                raise ValueError("Invalid data format")
            value['timestamp'] = datetime.now().isoformat()
            self.knowledge[key] = value
            self.save_knowledge()
            # Sync with Redis
            self.redis_client.set(key, json.dumps(value))
        logging.info(f"Added/Updated key: {key} at {datetime.now()}")

    def delete(self, key: str):
        with self.lock:
            if key in self.knowledge:
                del self.knowledge[key]
                self.save_knowledge()
                # Remove from Redis
                self.redis_client.delete(key)
                logging.info(f"Deleted key: {key} at {datetime.now()}")

    def search(self, query: str, threshold: int = 80) -> List[str]:
        with self.lock:
            return [key for key in self.knowledge if fuzz.partial_ratio(query.lower(), key.lower()) >= threshold]

    def get_all_keys(self) -> List[str]:
        with self.lock:
            return list(self.knowledge.keys())

    def create_backup(self):
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        backup_file = os.path.join(self.backup_dir, f"knowledge_base_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        shutil.copy2(self.file_path, backup_file)
        logging.info(f"Backup created: {backup_file}")

    def run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

    def validate_data(self, data: Any) -> bool:
        # Implement your data validation logic here
        return isinstance(data, dict) and 'content' in data

    def is_data_expired(self, data: Dict[str, Any]) -> bool:
        if 'timestamp' not in data:
            return False
        timestamp = datetime.fromisoformat(data['timestamp'])
        return (datetime.now() - timestamp) > timedelta(days=30)  # Expire after 30 days

class KnowledgeBaseManager:
    def __init__(self):
        self.kb = KnowledgeBase()

    def query(self, key: str) -> Dict[str, Any]:
        result = self.kb.get(key)
        if result:
            return {"found": True, "data": result}
        else:
            return {"found": False, "data": None}

    def add_or_update(self, key: str, value: Any) -> Dict[str, bool]:
        try:
            self.kb.set(key, value)
            return {"success": True}
        except Exception as e:
            logging.error(f"Error adding/updating key {key}: {str(e)}")
            return {"success": False, "error": str(e)}

    def remove(self, key: str) -> Dict[str, bool]:
        try:
            self.kb.delete(key)
            return {"success": True}
        except Exception as e:
            logging.error(f"Error removing key {key}: {str(e)}")
            return {"success": False, "error": str(e)}

    def search_knowledge(self, query: str) -> List[str]:
        return self.kb.search(query)

    def get_all_knowledge_keys(self) -> List[str]:
        return self.kb.get_all_keys()

def integrate_knowledge_base():
    import command_data
    import commands
    import help_info
    
    kb_manager = KnowledgeBaseManager()
    
    # Add KB-related commands
    command_data.COMMANDS['kb_query'] = {
        'function': lambda args: kb_manager.query(args[0]),
        'description': 'Query the knowledge base'
    }
    command_data.COMMANDS['kb_add'] = {
        'function': lambda args: kb_manager.add_or_update(args[0], {'content': args[1]}),
        'description': 'Add or update an entry in the knowledge base'
    }
    command_data.COMMANDS['kb_search'] = {
        'function': lambda args: kb_manager.search_knowledge(args[0]),
        'description': 'Search the knowledge base'
    }
    
    # Update help info
    help_info.HELP_INFO['kb_query'] = 'Usage: kb_query <key>'
    help_info.HELP_INFO['kb_add'] = 'Usage: kb_add <key> <value>'
    help_info.HELP_INFO['kb_search'] = 'Usage: kb_search <query>'
    
    # Modify existing command processing in commands.py to use KB when appropriate
    original_process_command = commands.process_command
    def new_process_command(command, args):
        if command in ['kb_query', 'kb_add', 'kb_search']:
            return command_data.COMMANDS[command]['function'](args)
        else:
            # Check KB first before processing normally
            kb_result = kb_manager.query(command)
            if kb_result['found']:
                return kb_result['data']['content']
            else:
                return original_process_command(command, args)
    
    commands.process_command = new_process_command

# Usage example (can be commented out if not needed)
if __name__ == "__main__":
    kb_manager = KnowledgeBaseManager()
    kb_manager.add_or_update("example_key", {"content": "This is an example value"})
    print(kb_manager.query("example_key"))
    print(kb_manager.search_knowledge("example"))