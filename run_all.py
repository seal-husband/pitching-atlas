"""
Run every script in order and log what each prints. 00 (fetching data) is not run here; see README.

Stops at the first script that fails. Output goes to output/run_all.log.

Usage:  python run_all.py            # everything
        python run_all.py 08 15      # only these, in the order given
"""
import os, sys, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["01_pitch_plane.py"], ["02_plane_height.py"], ["03_stuff_standin.py"], ["04_pitch_table.py"],
    ["05_peaks.py"], ["06_valleys.py"], ["07_four_numbers.py"], ["08_map.py"], ["09_separation.py"],
    ["10_reliability.py"], ["11_map_vs_stuff.py"], ["12_stability.py"], ["13_projection_vs_raw.py"],
    ["14_offplane_check.py"], ["15_note_figures.py"], ["16_cloud_data.py"],
]


def main():
    only = sys.argv[1:]
    steps = [s for s in STEPS if not only or s[0][:2] in only]
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    log = open(os.path.join(ROOT, "output", "run_all.log"), "w", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", LOKY_MAX_CPU_COUNT=os.environ.get("LOKY_MAX_CPU_COUNT", "4"))
    t_all = time.time()
    for step in steps:
        t = time.time()
        head = "===== %s" % " ".join(step)
        print(head, flush=True); log.write(head + "\n"); log.flush()
        r = subprocess.run([sys.executable, os.path.join("src", step[0])] + step[1:], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        log.write(r.stdout)
        if r.returncode:
            log.write(r.stderr)
        msg = "      exit %d  %.0f s" % (r.returncode, time.time() - t)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if r.returncode:
            print(r.stderr[-2000:]); sys.exit(r.returncode)
    print("all done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
