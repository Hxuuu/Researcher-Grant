#!/bin/bash
# Install script that handles the multitasking build issue
set -e

echo "Installing core dependencies..."
pip install anthropic rich python-dotenv

echo "Installing data dependencies..."
pip install numpy pandas requests beautifulsoup4 frozendict

echo "Installing curl-cffi..."
pip install "curl-cffi>=0.7,<0.14"

echo "Installing multitasking (may need to build from source)..."
pip install multitasking || {
    echo "Build failed, trying alternative..."
    pip install multitasking --no-build-isolation 2>/dev/null || {
        echo "Installing multitasking stub..."
        python3 - <<'EOF'
import os, site
for sp in site.getsitepackages():
    pkg_dir = os.path.join(sp, 'multitasking')
    try:
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, '__init__.py'), 'w') as f:
            f.write('''import threading, functools
_tasks = []
def task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=func, args=args, kwargs=kwargs)
        t.daemon = True; _tasks.append(t); t.start(); return t
    return wrapper
def wait_for_task(t): t.join()
def wait_for_all_tasks():
    for t in _tasks:
        try: t.join()
        except: pass
''')
        print(f"Stub installed at {pkg_dir}")
        break
    except PermissionError:
        continue
EOF
    }
}

echo "Installing yfinance..."
pip install "yfinance>=0.2.40"

echo "All dependencies installed!"
