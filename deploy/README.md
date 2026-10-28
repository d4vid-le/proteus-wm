# Deployment helpers

The reflex is a long-running surprise watcher. It is safe to run in the
foreground:

```bash
python -m proteus.agent.reflex --std-mult 5 --correct
```

For an unattended run on macOS, copy `com.proteus.reflex.plist.example`, replace
`__REPO__` with the absolute path to this repository, and load it:

```bash
sed "s|__REPO__|$PWD|g" deploy/com.proteus.reflex.plist.example \
  > ~/Library/LaunchAgents/com.proteus.reflex.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.proteus.reflex.plist
```

Unload with:

```bash
launchctl bootout gui/$(id -u)/com.proteus.reflex
```

The reflex defaults to dry-run. `--execute` is required for real corrective
mutations. See `docs/research/REFLEX.md` for the safety model.
