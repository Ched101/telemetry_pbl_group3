import subprocess
import sys


def run_command(command):
    print("\nRunning:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        print("Command failed.")
        sys.exit(result.returncode)


def main():
    run_command([sys.executable, "src/modulation_lead/am_modulation.py"])
    run_command([sys.executable, "src/modulation_lead/fm_modulation.py"])
    run_command([sys.executable, "src/modulation_lead/digital_modulation.py", "--scheme", "ASK"])
    run_command([sys.executable, "src/modulation_lead/digital_modulation.py", "--scheme", "FSK"])
    run_command([sys.executable, "src/modulation_lead/digital_modulation.py", "--scheme", "PSK"])


if __name__ == "__main__":
    main()