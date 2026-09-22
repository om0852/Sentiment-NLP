import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Sentiment Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI production server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable dev auto-reload")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model on processed dataset")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run CPU & RAM latency benchmark")

    # Load test command
    load_parser = subparsers.add_parser("load-test", help="Run traffic load simulation")
    load_parser.add_argument("--volume", type=int, default=1000000, help="Target posts/day (default: 1,000,000)")
    load_parser.add_argument("--duration", type=int, default=3, help="Duration in seconds (default: 3)")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run pytest automated test suite")

    args = parser.parse_args()

    # Find python executable in venv if present, otherwise system python
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(project_root, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    if args.command == "serve":
        cmd = [
            venv_python, "-m", "uvicorn", "src.serving.app:app",
            "--host", args.host,
            "--port", str(args.port),
            "--workers", "1"
        ]
        if args.reload:
            cmd.append("--reload")
        print(f"Starting server on {args.host}:{args.port}...")
        subprocess.run(cmd, cwd=project_root)

    elif args.command == "train":
        script = os.path.join(project_root, "src", "training", "train.py")
        subprocess.run([venv_python, script], cwd=project_root)

    elif args.command == "benchmark":
        script = os.path.join(project_root, "src", "benchmark", "benchmark_cpu_ram.py")
        subprocess.run([venv_python, script], cwd=project_root)

    elif args.command == "load-test":
        script = os.path.join(project_root, "src", "benchmark", "load_test.py")
        subprocess.run([venv_python, script], cwd=project_root)

    elif args.command == "test":
        pytest_exe = os.path.join(os.path.dirname(venv_python), "pytest.exe")
        if not os.path.exists(pytest_exe):
            pytest_exe = "pytest"
        subprocess.run([pytest_exe, "tests"], cwd=project_root)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
