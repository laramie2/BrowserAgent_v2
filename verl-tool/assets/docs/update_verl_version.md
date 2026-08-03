## Updating the bundled verl implementation

The localized `verl` implementation is tracked directly by the BrowserAgent v2
repository. It is intentionally not a Git submodule, because verl-tool carries
project-specific changes that must be reviewed and released together with the
rest of BrowserAgent.

To incorporate a newer upstream verl release:

1. clone the target upstream verl revision into a temporary directory;
2. review and port the upstream changes into `verl-tool/verl/`;
3. preserve the BrowserAgent-specific agent loop and configuration changes;
4. copy any required trainer defaults into `verl-tool/verl_tool/trainer/config/`;
5. run the BrowserAgent and verl-tool test suites; and
6. commit the resulting files in the BrowserAgent v2 repository.

After updating, reinstall the bundled package from `verl-tool/`:

```bash
uv pip install -e verl
```

Ensure `verl-tool/verl_tool/trainer/config/ppo_trainer.yaml` retains the
verl-tool agent default:

```yaml
defaults:
  - verltool@actor_rollout_ref.agent: agent.yaml
```
