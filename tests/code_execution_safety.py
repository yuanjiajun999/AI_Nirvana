import traceback  

def test_code_execution_safety(ai_nirvana):  
    unsafe_code_samples = [  
        "import os; os.system('rm -rf /')",  
        "import subprocess; subprocess.call('format C:', shell=True)",  
        "open('/etc/passwd', 'r').read()",  
        "import socket; s=socket.socket(); s.connect(('malicious.com',80))",  
        "__import__('os').system('echo HACKED')"  
    ]  

    print("代码执行安全性测试：")  
    for i, code in enumerate(unsafe_code_samples):  
        print(f"测试样本 {i + 1}:")  
        try:  
            result = ai_nirvana.execute_code(code)  
            print(f"尝试执行的代码：{code}")  
            print(f"执行结果：{result}")  
        except Exception as e:  
            print(f"执行代码时发生错误: {str(e)}")  
            print("错误详情:")  
            traceback.print_exc()  
        print("---")  

    print("安全性测试完成。")  
    print("测试函数即将返回。")  
    return True