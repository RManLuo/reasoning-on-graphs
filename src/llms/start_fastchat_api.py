import subprocess
import argparse
import sys
import atexit
import signal
processes = []

def terminate_process():
    for p in processes:
        print("Terminating process: ", p.pid)
        p.kill()
        p.wait()
        
    # Clean up log file.
    subprocess.call("rm model_worker_*.log", shell=True)
    subprocess.call("rm controller.log", shell=True)

def start_fastchat_api(model_names, model_path, conv_template, host, port):
    
    processes.append(subprocess.Popen(["python", "-m", "fastchat.serve.controller"], stdout = subprocess.PIPE,
                  stderr = subprocess.STDOUT, universal_newlines=True, bufsize=1))
    processes.append(subprocess.Popen(["python", "-m", "fastchat.serve.model_worker", "--model-path", model_path, "--model-names", model_names, "--conv-template", conv_template], stdout = subprocess.PIPE,
                  stderr = subprocess.STDOUT, universal_newlines=True))
    processes.append(subprocess.Popen(["python", "-m", "fastchat.serve.openai_api_server", "--host", host, "--port", str(port)], stdout = subprocess.PIPE,
                  stderr = subprocess.STDOUT))
            
    # Register signal handler
    atexit.register(terminate_process)
    print("Fastchat API starting....")
    # Check server status
    while True:
        output_0 = processes[0].stdout.readline()
        output_1 = processes[1].stdout.readline()
        print(output_1)
        if "Register done" in output_0:
            print("Fastchat API started.")
            break
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    argparser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    argparser.add_argument("--conv_template", type=str, default="llama-2")
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--port", type=int, default=8000)

    args = argparser.parse_args()
    start_fastchat_api(args.model_name, args.model_path, args.conv_template, args.host, args.port)
    def exit_handler(signal, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    while True:
        pass