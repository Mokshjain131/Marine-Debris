import subprocess
import sys


def run_script(script, args=None):
    cmd = [sys.executable, script]
    if args:
        cmd += args
    print(f"\n[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed.")
        sys.exit(result.returncode)


def main():
    # Step 1: Run evaluation.py to generate predictions
    print("\n=== Step 1: Evaluating model on test set ===")
    run_script("evaluation.py")

    # Step 2: Generate geolocation mapping from predictions
    print("\n=== Step 2: Generating geolocation map ===")
    run_script("utils/generate_geolocation_map_v2.py", ["--format", "detailed"])

    # Step 3: Generate world map HTML (all patches)
    print("\n=== Step 3: Generating world map HTML ===")
    run_script("extension.py", ["--per_patch", "--skip_route"])

    # Step 4: Generate route map HTML (optimal route)
    print("\n=== Step 4: Generating route map HTML ===")
    run_script("extension.py", ["--per_patch"])

    print("\n=== Pipeline Complete! ===")
    print("Output files:")
    print(" - marine_debris_world_map.html")
    print(" - marine_debris_route_map.html")


if __name__ == "__main__":
    main()
