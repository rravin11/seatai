# SeatAI development workflow

This repository uses small, documented, pushed implementation slices. The
objective is recoverability: the Git remote must always contain a coherent,
buildable explanation of the current project state.

## Required completion sequence

At each completed feature, bug fix, investigation conclusion, or roughly 200
net lines of code—whichever comes first:

1. Update the nearest README and the relevant technical document with the
   purpose, design decision, configuration change, limitations, and how to
   validate it. Add a dated entry to `docs/CHANGELOG.md` for material changes.
2. Build and run focused validation appropriate to the change. For camera work,
   run the capture probe before the full application. For live runtime changes,
   record whether each camera reached a first inference result.
3. Review `git status` and `git diff --check`. Do not stage unrelated user
   edits, raw imagery, generated reports/events, build trees, TensorRT plans,
   model weights, credentials, or virtual environments.
4. Make one imperative, scoped commit. Include validation in the commit body
   when it materially helps future diagnosis.
5. Push immediately with `git push origin main` and confirm the remote branch
   contains the new commit.

If a change is incomplete or a test is blocked by hardware, commit only the
safe, documented diagnostic work and state the outstanding condition plainly.
Never represent an unverified camera, model, or occupancy behavior as working.

## Documentation standard

Documentation should be technical and concise. Each material change should say:

- **Why:** problem or product need.
- **What:** components, interfaces, configuration, and data flow changed.
- **Verification:** exact command or test plus observed result.
- **Limits:** what remains unproven, unsafe, model-dependent, or deferred.

Keep stable architecture in `docs/TECHNICAL_STATE.md` and component documents.
Keep chronological decisions in `docs/CHANGELOG.md`. Avoid placing live model
metrics, secrets, personally identifiable imagery, or generated output in docs
that are committed to Git.

## Local commands

```bash
# Live product
cd seatvision
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/seatvision_capture_probe 0 60
./build/seatvision_capture_probe 1 60
./build/seatvisiond --config config/jetson.yaml

# Offline still-image lab
cd ../seatvision-dataset
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

# Publish a completed slice from repository root
cd ..
git status --short
git diff --check
git add <only-the-relevant-paths>
git commit -m "<scoped change>"
git push origin main
```
