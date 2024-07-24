import subprocess
import sys
import chardet
import logging
from datetime import datetime
import yagmail

# 设置日志
log_filename = f"dependency_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def read_file_with_encoding(file_path):
    encoding = detect_file_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def run_command(command):
    logging.info(f"Running command: {command}")
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    logging.info(f"Command output: {result.stdout}")
    if result.stderr:
        print("Error:", result.stderr, file=sys.stderr)
        logging.error(f"Command error: {result.stderr}")
    return result.returncode

def update_dependencies():
    return run_command("python update_all_requirements.py")

def run_safety_check():
    requirements = read_file_with_encoding("requirements.txt")
    process = subprocess.Popen(["safety", "check", "--stdin"], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
    stdout, stderr = process.communicate(input=requirements)
    print(stdout)
    logging.info(f"Safety check output: {stdout}")
    if stderr:
        print("Error:", stderr, file=sys.stderr)
        logging.error(f"Safety check error: {stderr}")
    return process.returncode, stdout

def update_documentation(check_results):
    with open("SECURITY.md", "a") as doc:
        doc.write(f"\n\n## Security Check - {datetime.now().strftime('%Y-%m-%d')}\n")
        doc.write("Results of the latest security check:\n")
        doc.write(check_results)
    logging.info("Documentation updated with security check results")

def send_notification(subject, body):
    try:
        yag = yagmail.SMTP("yuanjiajun999@gmail.com", "qq215512")
        yag.send(to="yuanjiajun999@gmail.com", subject=subject, contents=body)
        logging.info(f"Notification sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send notification: {str(e)}")

def main():
    logging.info("Starting dependency management and security check")
    
    # 更新依赖
    if update_dependencies() != 0:
        error_msg = "Error updating dependencies"
        print(error_msg)
        logging.error(error_msg)
        send_notification("Dependency Update Failed", error_msg)
        return

    # 运行安全检查
    check_code, check_results = run_safety_check()
    if check_code != 0:
        msg = "Security vulnerabilities detected. Please review the output above."
        print(msg)
        logging.warning(msg)
        update_documentation(check_results)
        send_notification("Security Vulnerabilities Detected", check_results)
    else:
        msg = "No security vulnerabilities detected."
        print(msg)
        logging.info(msg)

    print("Dependency management and security check completed.")
    logging.info("Dependency management and security check completed")
    print(f"Log file created: {log_filename}")

if __name__ == "__main__":
    main()